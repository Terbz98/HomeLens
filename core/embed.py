"""Turn an exterior photo into a list of numbers (an embedding). Three encoders are compared.

- "resnet18": torchvision ResNet18, ImageNet weights, 512 numbers.
- "convnext_tiny": torchvision ConvNeXt-Tiny, ImageNet weights, 768 numbers. A newer
  and stronger image network than ResNet18.
- "harris": hand-made corner and edge features with the Harris method of DSP Lab 7,
  36 numbers (see core/harris.py).

The networks are used as fixed feature extractors: HomeLens does not train or
change them. torchvision code is BSD-3-Clause. The weights were trained on
ImageNet, whose images are for non-commercial research and education. See
docs/licensing.md.

torch is imported only when an embedding is actually needed, so the backend
starts fast in location-only mode.
"""
import io
from typing import Dict, List

import numpy as np

ENCODERS: Dict[str, Dict] = {
    "resnet18": {"name": "torchvision resnet18 IMAGENET1K_V1", "size": 512},
    "convnext_tiny": {"name": "torchvision convnext_tiny IMAGENET1K_V1", "size": 768},
    "harris": {"name": "Harris corner features (DSP Lab 7 method), 3 x 3 grid", "size": 36},
}
DEFAULT_ENCODER = "resnet18"
# Kept for older callers: the name and size of the default encoder.
ENCODER_NAME = ENCODERS[DEFAULT_ENCODER]["name"]
EMBEDDING_SIZE = ENCODERS[DEFAULT_ENCODER]["size"]

_models: Dict[str, tuple] = {}


def _load(encoder: str):
    if encoder not in _models:
        import torch
        if encoder == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()          # keep the 512 numbers before the class layer
        elif encoder == "convnext_tiny":
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            model = convnext_tiny(weights=weights)
            model.classifier[2] = torch.nn.Identity()  # keep the 768 numbers before the class layer
        else:
            raise ValueError("unknown encoder %r" % encoder)
        model.eval()
        _models[encoder] = (model, weights.transforms())
    return _models[encoder]


def embed_images(jpeg_bytes_list: List[bytes], encoder: str = DEFAULT_ENCODER) -> np.ndarray:
    """Return an array with one row per photo."""
    size = ENCODERS[encoder]["size"]
    if not jpeg_bytes_list:
        return np.zeros((0, size), dtype=np.float32)
    if encoder == "harris":
        from core import harris
        return harris.embed_images(jpeg_bytes_list)
    import torch
    from PIL import Image

    model, preprocess = _load(encoder)
    tensors = [preprocess(Image.open(io.BytesIO(data)).convert("RGB")) for data in jpeg_bytes_list]
    with torch.no_grad():
        out = model(torch.stack(tensors))
    return out.numpy().astype(np.float32).reshape(-1, size)


def embed_image(jpeg_bytes: bytes, encoder: str = DEFAULT_ENCODER) -> np.ndarray:
    """Embedding of a single photo, shape (size,)."""
    return embed_images([jpeg_bytes], encoder)[0]
