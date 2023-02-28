import pytest
import pickle
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from clip_retrieval.clip_inference.mapper import ISCMapper,SSCDMapper

PILtrans = transforms.ToPILImage()

def test_mapper():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # mapper = ClipMapper(
    #     enable_image=True,
    #     enable_text=False,
    #     enable_metadata=False,
    #     use_mclip=False,
    #     clip_model=model,
    #     use_jit=True,
    #     mclip_model="",
    # )

    mapper = SSCDMapper(
        enable_image=True,enable_metadata=False,enable_text=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tensor_files = [i for i in os.listdir(current_dir + "/test_tensors")]

    for tensor_file in tensor_files:
        with open(current_dir + "/test_tensors/{}".format(tensor_file), "rb") as f:
            tensor = pickle.load(f)
            sample = mapper(tensor)
            assert sample["image_embs"].shape[0] == tensor["image_tensor"].shape[0]
            assert sample["image_embs"].dtype == np.dtype("float16")
        pass
