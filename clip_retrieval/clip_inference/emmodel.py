from torchvision import models
from torch import nn
import torch
from torchvision.transforms import transforms
from clip_retrieval.clip_inference.iscm import create_iscmodel
class MobilenetV3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=True).eval()
        self.mobilenet_gap_op = torch.nn.Sequential(
            mobilenet.features, mobilenet.avgpool
        )

    def forward(self, x) -> torch.tensor:
        return self.mobilenet_gap_op(x)


def create_model_mobilenet():
    model = MobilenetV3()
    preprocessor = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return model, preprocessor


def create_model_isc():
    model, preprocessor = create_iscmodel(weight_name='isc_ft_v107', device='cuda')
    return model, preprocessor


def create_model_sscd():
    model = torch.jit.load("/admin/home-harrysaini/models/sscd_disc_mixup.torchscript.pt")
    preprocessor = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
    ])
    return model,preprocessor


