"""
This is an example on how to use embedding reader to do an inference over a set of billion
of clip vit-l/14 embeddings to predict tags for each example, representing the aesthetic value of images
"""


from builtins import ValueError
from embedding_reader import EmbeddingReader
import fire
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import fsspec
import math
import pandas as pd
import pyarrow.parquet as pq


dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


import mmh3


def compute_hash(url, text):
    if url is None:
        url = ""

    if text is None:
        text = ""

    total = (url + text).encode("utf-8")
    return mmh3.hash64(total)[0]


def main(
    embedding_folder="data/bm25_emb_test/text_emb/",
    metadata_folder="data/bm25_emb_test/metadata/",
    output_folder="data/bm25_emb_test/final",
    batch_size=10**6,
    end=None,
        bm25_k=1.2,
        bm25_b=0.75,
        stats_folder='data/bm25_emb_test/'
):
    """main function"""
    reader = EmbeddingReader(
        embedding_folder, metadata_folder=metadata_folder, file_format="parquet_npy", meta_columns=["url", "caption",'doc_lens']
    )
    fs, relative_output_path = fsspec.core.url_to_fs(output_folder)

    emb_dir = os.path.join(relative_output_path,'text_emb')
    meta_dir = os.path.join(relative_output_path, 'metadata')
    fs.mkdirs(relative_output_path, exist_ok=True)

    stats_fname = os.path.join(stats_folder,'text_stats.npz')
    stats = np.load(stats_fname)
    idf = stats['idf']
    avg_len = stats['avg_len']

    total = reader.count
    batch_count = max(1,math.ceil(total // batch_size))
    padding = int(math.log10(batch_count)) + 1

    for i, (emb, ids) in enumerate(reader(batch_size=batch_size, start=0, end=end)):
        # doc_lens = emb.sum(axis=1, keepdims=True)
        doc_lens = np.asarray(ids['doc_lens'])[:,None]
        emb = (emb * (bm25_k + 1)) / (emb + bm25_k * (1 - bm25_b + bm25_b * doc_lens / avg_len)) * idf[np.newaxis]


        padded_id = str(i).zfill(padding)
        output_file_path = os.path.join(relative_output_path, padded_id + ".parquet")
        df = pd.DataFrame.from_dict(ids)
        df["hash"] = [compute_hash(x, y) for x, y in zip(ids["url"], ids["caption"])]
        with fs.open(output_file_path, "wb") as f:
            df.to_parquet(f)

        np.save(os.path.join(relative_output_path,f'embeddings_bm25_{padded_id}.npy'),emb)


if __name__ == "__main__":
    fire.Fire(main)
