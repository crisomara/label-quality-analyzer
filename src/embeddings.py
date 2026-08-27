"""Frozen pretrained-CNN embedding extraction (CPU-only, by design).

Rather than fine-tuning a CNN end-to-end on CIFAR-10 (the approach that caused
an out-of-memory crash on this same Windows machine during the music-recommender
project), we do a single frozen forward pass per image through an ImageNet-
pretrained MobileNetV2 and keep the 1280-dimensional pooled feature vector. Confident
learning only needs well-calibrated predicted probabilities, and those come
from training a cheap classifier (logistic regression) on top of these frozen
embeddings — no GPU, no backprop through the CNN, no long training loop.

**Resize, don't feed native 32x32.** A quick diagnostic (4,000-image subsample,
5-fold cross-validated logistic regression on top of the frozen features) showed
feeding CIFAR-10's native 32x32 images straight into MobileNetV2 gives poor
features -- 50.2% accuracy -- because the network's BatchNorm statistics were
computed at 224x224, and the size mismatch skews activations badly. Simply
resizing images up before the forward pass (no fine-tuning, still fully frozen,
still CPU-only) recovers most of that:

| resize | accuracy (4k subsample) | extraction time (60k images, CPU) |
|---|---|---|
| 32x32 (native, no resize) | 50.2% | ~1 min |
| 96x96  | 75.1% | ~7 min |
| 128x128 | 79.1% | ~12 min |
| 224x224 (full ImageNet size) | 82.8% | ~39 min |

128x128 was chosen as the default: it captures most of the accuracy gain 224x224
gives, at roughly a third of the compute cost -- a reasonable speed/quality
trade-off for a CPU-only pipeline audited over 60,000 images.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_RESIZE = 128


def _build_transform(resize: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize), antialias=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class _ArrayImageDataset(Dataset):
    """Wraps a torchvision CIFAR10 dataset's raw numpy image array for a plain DataLoader."""

    def __init__(self, images: np.ndarray, resize: int = DEFAULT_RESIZE):
        self.images = images  # (N, 32, 32, 3) uint8
        self.transform = _build_transform(resize)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])


def load_frozen_mobilenet() -> torch.nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_embeddings(
    images: np.ndarray,
    model: torch.nn.Module,
    batch_size: int = 256,
    num_workers: int = 0,
    resize: int = DEFAULT_RESIZE,
    progress_cb=None,
) -> np.ndarray:
    """Run a frozen forward pass over `images` (N,32,32,3 uint8), return (N,1280) float32.

    Each image is resized to `resize`x`resize` before the forward pass -- see the
    module docstring for why native 32x32 was rejected in favor of this."""
    ds = _ArrayImageDataset(images, resize=resize)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out = np.empty((len(images), 1280), dtype=np.float32)
    ptr = 0
    for batch in loader:
        feats = model.features(batch)  # (B, 1280, H', W')
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
        n = pooled.shape[0]
        out[ptr:ptr + n] = pooled.numpy()
        ptr += n
        if progress_cb is not None:
            progress_cb(ptr, len(images))
    return out


def cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name
