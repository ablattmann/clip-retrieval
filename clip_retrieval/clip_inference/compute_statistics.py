"""
This is an example on how to use embedding reader to do an inference over a set of billion
of clip vit-l/14 embeddings to predict tags for each example, representing the aesthetic value of images
"""


from builtins import ValueError
from embedding_reader import EmbeddingReader
import fire
import os
import numpy as np

import fsspec
import math
import pandas as pd
import pyarrow.parquet as pq


dir_path = os.path.dirname(os.path.realpath(__file__))



def main(
    embedding_folder="data/bm25_emb_test/text_emb/",
    metadata_folder="data/bm25_emb_test/metadata/",
    output_folder="/data/bm25_emb_test",
    batch_size=10**5,
    vocab_size=32100,
    end=None,
):
    """main function"""
    reader = EmbeddingReader(
        embedding_folder, metadata_folder=metadata_folder, file_format="parquet_npy", meta_columns=["url", "caption", "doc_lens"]
    )
    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)
    fs.mkdirs(relative_output_path, exist_ok=True)

    # model = get_aesthetic_model()
    total = reader.count
    # batch_count = max(1,math.ceil(total // batch_size))
    doc_occ_per_token = np.zeros((vocab_size,), dtype=np.float16)
    doc_lens = None

    for i, (embeddings, ids) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        occs_in_batch = np.clip(embeddings, 0, 1).sum(0)
        if doc_lens is None:
            doc_lens = np.asarray(ids['doc_lens'])
        else:
            doc_lens = np.append(doc_lens,ids['doc_lens'])

        doc_occ_per_token += occs_in_batch


    n_docs = total if end is None else end
    idf = np.log((n_docs - doc_occ_per_token + 0.5) / (doc_occ_per_token + 0.5) + 1)
    avg_dlen = doc_lens.mean()

    savepath = os.path.join(output_folder,'text_stats.npz')
    np.savez(savepath,idf=idf,avg_len=avg_dlen)

    print('done')



if __name__ == "__main__":
    fire.Fire(main)
