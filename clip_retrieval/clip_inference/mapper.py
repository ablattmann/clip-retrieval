"""mapper module transform images and text to embeddings"""

import torch
import numpy as np
from clip_retrieval.load_clip import load_clip
from sentence_transformers import SentenceTransformer
from abc import abstractmethod
from clip_retrieval.clip_inference.text_mappers import __TEXT_EMBEDDERS__
from clip_retrieval.clip_inference.emmodel import create_model_isc,create_model_sscd,create_model_mobilenet

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class BaseMapper:

    def __init__(self, enable_image, enable_text, enable_metadata,*args, **kwargs):
        self.enable_image = enable_image
        self.enable_text = enable_text
        self.enable_metadata = enable_metadata

    @abstractmethod
    def __call__(self, item):
        raise NotImplementedError('base class should never be called')

# TODO fix this
class BM25Mapper(BaseMapper):
    """transforms images and texts into embeddings for BM25-based retrieval"""
    def __init__(self,
                 clip_model,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.enable_image, f'{self.__class__.__name__} is a text-only mapping'
        self.model = __TEXT_EMBEDDERS__[clip_model](**kwargs)

        print(f'Load {self.model.__class__.__name__} with vocab size {self.model.vocab_size}')


    def __call__(self, item):
        image_embs = None
        metadata = item["metadata"]

        with torch.no_grad():
            text_tokens, doc_lens = self.model.encode(item['text'],return_lengths=True)

        # TODO this assumes that metadata is a dict
        bow = []

        for i, (tt, dl) in enumerate(zip(text_tokens,doc_lens)):
            # metadata[i].update({'doc_len': dl})
            bow.append(np.bincount(np.asarray(tt),minlength=self.model.vocab_size).astype(np.float16))

        bow = np.stack(bow,axis=0)

        # metadata['doc_len'] = doc_lens.cpu().numpy().astype(int)
        #
        # bow = np.apply_along_axis(np.bincount, 1, text_tokens.cpu().numpy(), minlength=self.model.vocab_size).astype(int)
        # occs_in_batch = np.clip(bow, 0, 1).sum(0).astype(int)

        # TODO this assumes that metadata is a dict this will likely fail
        # metadata['occurence_per_token'] = occs_in_batch
        text = item['text']

        return {
            "image_embs": image_embs,
            "text_embs": bow,
            "image_filename": None,
            "text": text,
            "metadata": metadata,
            'doc_lens': doc_lens
        }

class SentenceTransformerMapper(BaseMapper):
    """Transforms text into embeddings of a given sentence transformers model as specified by strans_model"""
    def __init__(self,
                strans_model,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.enable_image, f'{self.__class__.__name__} is a text-only mapping'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Loading {strans_model} from SentenceTransformers for creating embeddings')
        model = SentenceTransformer(strans_model)
        self.model = model.encode

    @torch.no_grad()
    def __call__(self, item):
        text = None
        text_embs = None
        metadata = None
        if self.enable_text:
            text_embs = normalized(self.model(item["text"])).astype(np.float16)
            text = item['text']

        if self.enable_metadata:
            metadata = item['metadata']

        return {
            "image_embs": None,
            "text_embs": text_embs,
            "image_filename": None,
            "text": text,
            "metadata": metadata,
        }





class ClipMapper(BaseMapper):
    """transforms images and texts into clip embeddings"""

    def __init__(
        self,
        use_mclip,
        clip_model,
        use_jit,
        mclip_model,
        *args,
        warmup_batch_size=1,
        clip_cache_path=None,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        self.use_mclip = use_mclip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_clip(
            clip_model=clip_model, use_jit=use_jit, warmup_batch_size=warmup_batch_size, clip_cache_path=clip_cache_path
        )
        self.model_img = model.encode_image
        self.model_txt = model.encode_text
        if use_mclip:
            print("\nLoading MCLIP model for text embedding\n")
            mclip = SentenceTransformer(mclip_model)
            self.model_txt = mclip.encode

    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs = None
            image_filename = None
            text = None
            metadata = None
            if self.enable_image:
                image_features = self.model_img(item["image_tensor"].to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().to(torch.float16).numpy()
                image_filename = item["image_filename"]
            if self.enable_text:
                if self.use_mclip:
                    text_embs = normalized(self.model_txt(item["text"]))
                else:
                    text_features = self.model_txt(item["text_tokens"].to(self.device))
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_embs = text_features.cpu().to(torch.float16).numpy()
                text = item["text"]
            if self.enable_metadata:
                metadata = item["metadata"]

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }

class ISCMapper(BaseMapper):
    # load an img encoder model
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model,self.preprocessor = create_model_isc()
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs = None
            image_filename = None
            text = None
            metadata = None
            if self.enable_image:
                image_features = self.model(item["image_tensor"].to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().to(torch.float16).numpy()
                image_filename = item["image_filename"]
            if self.enable_text:
                text = item["text"]
            if self.enable_metadata:
                metadata = item["metadata"]

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }

class MobileNetV3Mapper(BaseMapper):
    # load an img encoder model
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model,self.preprocessor = create_model_mobilenet()
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            metadata = None
            if self.enable_image:
                image_features = self.model(item["image_tensor"].to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().to(torch.float16).numpy()
                image_filename = item["image_filename"]
            if self.enable_text:
                text = item["text"]

            if self.enable_metadata:
                metadata = item["metadata"]

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }

class SSCDMapper(BaseMapper):
    # load an img encoder model
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #model, preprocessor = create_model(weight_name='isc_ft_v107', device='cuda')
        self.model,self.preprocessor = create_model_sscd()
        self.model.eval()
        self.model.to(self.device)


    def __call__(self, item):
        with torch.no_grad():
            image_embs = None
            text_embs = None
            image_filename = None
            text = None
            metadata = None
            if self.enable_image:
                image_features = self.model(item["image_tensor"].to(self.device))
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embs = image_features.cpu().to(torch.float16).numpy()
                image_filename = item["image_filename"]

            if self.enable_text:
                text = item["text"]
                
            if self.enable_metadata:
                metadata = item["metadata"]

            return {
                "image_embs": image_embs,
                "text_embs": text_embs,
                "image_filename": image_filename,
                "text": text,
                "metadata": metadata,
            }