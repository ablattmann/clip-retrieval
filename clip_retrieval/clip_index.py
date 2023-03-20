"""Clip index is a tool to index clip embeddings using autofaiss"""

import fire
import os
from distutils.dir_util import copy_tree
import logging


LOGGER = logging.getLogger(__name__)


def quantize(emb_folder, index_folder, index_name, max_index_memory_usage, current_memory_available, nb_cores, index_key):
    """calls autofaiss to build an index"""

    from autofaiss import build_index  # pylint: disable=import-outside-toplevel

    try:
        LOGGER.debug(f"starting index {index_name}")
        if os.path.exists(emb_folder):
            LOGGER.debug(
                f"embedding path exist, building index {index_name}"
                f"using embeddings {emb_folder} ; saving in {index_folder}"
            )
            build_index(
                embeddings=emb_folder,
                index_path=index_folder + "/" + index_name + ".index",
                index_infos_path=index_folder + "/" + index_name + ".json",
                max_index_memory_usage=max_index_memory_usage,
                current_memory_available=current_memory_available,
                nb_cores=nb_cores,
                index_key=index_key,
            )
            LOGGER.debug(f"index {index_name} done")
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.exception(f"index {index_name} failed")
        raise e


def clip_index(
    embeddings_folder,
    index_folder,
    max_index_memory_usage="4G",
    current_memory_available="16G",
    copy_metadata=True,
    image_subfolder="img_emb",
    text_subfolder="text_emb",
    nb_cores=None,
    index_key=None,
):
    """indexes clip embeddings using autofaiss"""

    if os.path.isdir(embeddings_folder + "/" + image_subfolder):
        quantize(
            embeddings_folder + "/" + image_subfolder,
            index_folder,
            "image",
            max_index_memory_usage,
            current_memory_available,
            nb_cores,
            index_key=index_key
        )
    else:
        print('skipping image index since no image embeddings')

    if os.path.isdir(embeddings_folder + '/' + text_subfolder):
        quantize(
            embeddings_folder + "/" + text_subfolder,
            index_folder,
            "text",
            max_index_memory_usage,
            current_memory_available,
            nb_cores,
            index_key=index_key
        )
    else:
        print('skipping text lindex since no text embeddings')
    if copy_metadata:
        copy_tree(embeddings_folder + "/metadata", index_folder + "/metadata")


if __name__ == "__main__":
    fire.Fire(clip_index)
