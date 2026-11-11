"""Turn an exterior photo into a list of 512 numbers (an embedding).

The encoder is ResNet18 from torchvision with its pretrained ImageNet
weights. It is used as a fixed feature extractor: HomeLens does not train
or change it. torchvision code is BSD-3-Clause. The weights were trained on
ImageNet, whose images are for non-commercial research and education. See
docs/licensing.md.

torch is imported only when an embedding is actually needed, so the backend
starts fast in location-only mode.
"""
import io
from typing import List

import numpy as np

ENCODER_NAME = "torchvision resnet18 IMAGENET1K_V1"
EMBEDDING_SIZE = 512

_model = None
_preprocess = None


def _load():
    global _model, _preprocess
    if _model is None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()  # keep the 512 numbers before the class layer
        model.eval()
        _model = model
        _preprocess = weights.transforms()
    return _model, _preprocess


def embed_images(jpeg_bytes_list: List[bytes]) -> np.ndarray:
    """Return an array with one 512-number row per photo."""
    import torch
    from PIL import Image

    model, preprocess = _load()
    tensors = []
    for data in jpeg_bytes_list:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensors.append(preprocess(image))
    if not tensors:
        return np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    with torch.no_grad():
        out = model(torch.stack(tensors))
    return out.numpy().astype(np.float32)


def embed_image(jpeg_bytes: bytes) -> np.ndarray:
    """Embedding of a single photo, shape (512,)."""
    return embed_images([jpeg_bytes])[0]
