"""heavily borrowed from https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_back.py. Thanks!"""
from typing import Union,List,ByteString

import numpy as np
from io import BytesIO
from PIL import Image
import base64
from fire import Fire
import faiss
import math
from collections import defaultdict
import time

from fastapi import FastAPI, APIRouter
import uvicorn
from pydantic import BaseModel


from clip_retrieval.clip_back import load_indices, EmbedderOptions, convert_metadata_to_base64, \
    meta_to_dict, normalized
from img2dataset.downloader import Downloader


#TODO only dummy for later , but should in general be optional (first look how to best send images)
class Query(BaseModel):
    text: List[str]
    knn: int = 4
    image: Union[str,None] = None
    use_img: bool = False
    aesthetic_weight: Union[float,None] = None
    aesthetic_score: Union[float,None] = None
    use_safety_model: bool = False
    use_violence_detector: bool = False
    dedup: bool = True



class KNNService(object):
    """knn service which can be triggered asynchronously via http"""

    def __init__(self,
                 indeces_paths,
                 embedder_options:EmbedderOptions):
        self.router = APIRouter()
        self.router.add_api_route('/knnservice/',self.search,methods=['POST'])
        print(f'{self.__class__.__name__} loading model and index')
        start = time.time()
        self.knn_resource = load_indices(indices_paths=indeces_paths,
                                         clip_options=embedder_options)
        print(f'Loading model and index took {time.time() - start} secs')


    def compute_query(
        self,
        clip_resource,
        text_input,
        image_input,
        image_url_input,
        use_mclip,
        aesthetic_score,
        aesthetic_weight,
        use_stformer
    ):
        """compute the query embedding"""
        import torch  # pylint: disable=import-outside-toplevel
        import clip  # pylint: disable=import-outside-toplevel

        if text_input is not None and text_input != "":
            if use_stformer:
                with torch.no_grad():
                    query = normalized(clip_resource.stformer_model(text_input))

            else:
                if use_mclip:
                    query = normalized(clip_resource.model_txt_mclip(text_input))
                else:
                    text = clip.tokenize([text_input], truncate=True).to(clip_resource.device)
                    with torch.no_grad():
                        text_features = clip_resource.model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    query = text_features.cpu().to(torch.float32).detach().numpy()
        elif image_input is not None or image_url_input is not None:
            # TODO
            raise NotImplementedError()
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            img = Image.open(img_data)
            prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
            with torch.no_grad():
                image_features = clip_resource.model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().to(torch.float32).detach().numpy()

        if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
            aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
            query = query + aesthetic_embedding * aesthetic_weight
            query = query / np.linalg.norm(query)

        return query

    def connected_components(self, neighbors):
        """find connected components in the graph"""
        seen = set()

        def component(node):
            r = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def get_non_uniques(self, embeddings, threshold=0.94):
        """find non-unique embeddings"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)  # pylint: disable=no-value-for-parameter
        l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

        same_mapping = defaultdict(list)

        # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
        for i in range(embeddings.shape[0]):
            for j in I[l[i] : l[i + 1]]:
                same_mapping[int(i)].append(int(j))

        groups = self.connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return list(non_uniques)

    def connected_components_dedup(self, embeddings):
        non_uniques = self.get_non_uniques(embeddings)
        return non_uniques

    def get_unsafe_items(self, safety_model, embeddings, threshold=0.5):
        """find unsafe embeddings"""
        nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
        x = np.array([e[0] for e in nsfw_values])
        return np.where(x > threshold)[0]

    def get_violent_items(self, safety_prompts, embeddings):
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]



    def post_filter(
        self, safety_model, embeddings, deduplicate, use_safety_model, use_violence_detector, violence_detector
    ):
        """post filter results : dedup, safety, violence"""
        to_remove = set()
        if deduplicate:
            to_remove = set(self.connected_components_dedup(embeddings))

        if use_violence_detector and violence_detector is not None:
            to_remove |= set(self.get_violent_items(violence_detector, embeddings))
        if use_safety_model and safety_model is not None:
            to_remove |= set(self.get_unsafe_items(safety_model, embeddings))

        return to_remove

    def knn_search(
        self, query, modality, num_result_ids, clip_resource, deduplicate,
        use_safety_model, use_violence_detector,reconstruct=False
    ):
        """compute the knn search"""

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index
        if clip_resource.metadata_is_ordered_by_ivf:
            ivf_old_to_new_mapping = clip_resource.ivf_old_to_new_mapping

        index = image_index if modality == "image" else text_index

        if clip_resource.metadata_is_ordered_by_ivf:
            previous_nprobe = faiss.extract_index_ivf(index).nprobe
            if num_result_ids >= 100000:
                nprobe = math.ceil(num_result_ids / 3000)
                params = faiss.ParameterSpace()
                params.set_index_parameters(index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}")
        if reconstruct:
            distances, indices, embeddings = index.search_and_reconstruct(query, num_result_ids)
        else:
            distances, indices = index.search(query, num_result_ids)

        if clip_resource.metadata_is_ordered_by_ivf:
            results = []
            for ids in indices:
                results.append(np.take(ivf_old_to_new_mapping, ids))
            results = np.stack(results)
        else:
            results = indices



        if clip_resource.metadata_is_ordered_by_ivf:
            params = faiss.ParameterSpace()
            params.set_index_parameters(index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}")

        if reconstruct:

            final_ids = []
            final_distances  = []
            for res, dist, emb in zip(results,distances,embeddings):

                nb_results = np.where(res == -1)[0]

                if len(nb_results) > 0:
                    nb_results = nb_results[0]
                else:
                    nb_results = len(res)
                result_indices = res[:nb_results]
                result_distances = dist[:nb_results]
                result_embeddings = emb[:nb_results]
                result_embeddings = normalized(result_embeddings)
                local_indices_to_remove = self.post_filter(
                    clip_resource.safety_model,
                    result_embeddings,
                    deduplicate,
                    use_safety_model,
                    use_violence_detector,
                    clip_resource.violence_detector,
                )
                indices_to_remove = set()
                for local_index in local_indices_to_remove:
                    indices_to_remove.add(result_indices[local_index])
                indices = []
                distances = []
                for ind, distance in zip(result_indices, result_distances):
                    if ind not in indices_to_remove:
                        indices_to_remove.add(ind)
                        indices.append(ind)
                        distances.append(distance)
                final_ids.append(indices)
                final_distances.append(distances)
        else:
            final_distances = None
            final_ids = results

        return final_distances, final_ids


    async def search(self, query:Query):

        knn_resource = self.knn_resource['test_index']

        if query.use_img:
            assert query.image is not None
            text_input =None
            img_input = query.image if isinstance(query.image,ByteString) else None
            img_url_input = query.image if isinstance(query.image,str) else None
            modality='image'
        else:
            text_input = query.text
            img_input = None
            img_url_input = None
            modality = 'text'
        start = time.time()
        query_emb = self.compute_query(knn_resource,text_input=text_input,
                                       image_input=img_input,image_url_input=img_url_input,
                                       use_mclip=False,aesthetic_score=query.aesthetic_score,
                                       aesthetic_weight=query.aesthetic_weight,
                                       use_stformer=knn_resource.use_stformer)
        print(f'query embeddings took {time.time() - start} secs')
        start = time.time()
        dists, ids = self.knn_search(query_emb,modality=modality,num_result_ids=query.knn,
                                     clip_resource=knn_resource,deduplicate=query.dedup,
                                     use_safety_model=query.use_safety_model,
                                     use_violence_detector=query.use_violence_detector,
                                     reconstruct=query.dedup)

        print(f'search took {time.time() - start} secs')

        return query

    def map_to_metadata(self, indices, distances, metadata_provider, columns_to_return):
        """map the indices to the metadata"""

        results = []
        for id_, dist in zip(indices,distances):
            metas = metadata_provider.get(id_, columns_to_return)
            row_results = []
            for key, (d, i) in enumerate(zip(dist, id_)):
                output = {}
                meta = None if key + 1 > len(metas) else metas[key]
                convert_metadata_to_base64(meta)
                if meta is not None:
                    output.update(meta_to_dict(meta))
                output["id"] = i.item()
                output["similarity"] = d.item()
                row_results.append(output)
            results.append(row_results)

        return results


def knn_service(index_path='indices_paths.json',
                enable_hdf5=False,
                enable_faiss_memory_mapping=False,
                columns_to_return=None,
                reorder_metadata_by_ivf_index=False,
                enable_mclip_option=True,
                clip_model="ViT-B/32",
                use_jit=True,
                use_arrow=False,
                provide_safety_model=False,
                provide_violence_detector=False,
                provide_aesthetic_embeddings=True,
                mapper_type='CLIP',
                stformer_model='sentence-transformers/all-mpnet-base-v2',
                pub_port=8000
                ):
    app = FastAPI()

    embedder_options = EmbedderOptions(indice_folder='',
                                       clip_model=clip_model,
                                       enable_hdf5=enable_hdf5,
                                       enable_faiss_memory_mapping=enable_faiss_memory_mapping,
                                       columns_to_return=columns_to_return,
                                       reorder_metadata_by_ivf_index=reorder_metadata_by_ivf_index,
                                       enable_mclip_option=enable_mclip_option,
                                       use_jit=use_jit,
                                       use_arrow=use_arrow,
                                       provide_safety_model=provide_safety_model,
                                       provide_violence_detector=provide_violence_detector,
                                       provide_aesthetic_embeddings=provide_aesthetic_embeddings,
                                       mapper_type=mapper_type,
                                       stformer_model=stformer_model
                                       )

    knn_provider = KNNService(index_path,embedder_options)
    app.include_router(knn_provider.router)
    uvicorn.run(app, port=pub_port)

if __name__ == '__main__':
    Fire(knn_service)
