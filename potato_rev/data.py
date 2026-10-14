import hashlib
import json
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from potato_rev import config as C


def load_manifest(path: str = None) -> dict:
    with open(path or C.MANIFEST_PATH, "r") as f:
        return json.load(f)


def _load_square(path: str, size: int) -> np.ndarray:
    """Resize the short side to `size` (bilinear, antialiased) and centre-crop to size x size."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    return np.asarray(im.crop((left, top, left + size, top + size)), dtype=np.uint8)


def build_cache(data_dir: str, manifest: dict, cache_dir: str) -> np.ndarray:
    """
    Decode every curated image once and store it as a uint8 array [N, S, S, 3].
    Each file's MD5 is checked against the manifest so that a different copy of the
    dataset cannot silently change the experiment.
    """
    kept = [m for m in manifest["images"] if m["status"] == "kept"]
    key = hashlib.md5("".join(m["md5"] for m in kept).encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"cache_{C.CACHE_SIZE}_{key}.npy")
    if os.path.isfile(cache_path):
        return np.load(cache_path)

    arr = np.empty((len(kept), C.CACHE_SIZE, C.CACHE_SIZE, 3), dtype=np.uint8)
    for i, m in enumerate(kept):
        path = os.path.join(data_dir, m["path"])
        with open(path, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        if md5 != m["md5"]:
            raise RuntimeError(f"MD5 mismatch for {m['path']}: the dataset differs from the audited copy.")
        arr[i] = _load_square(path, C.CACHE_SIZE)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = cache_path + f".tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, cache_path)
    return arr


class PotatoData:
    """The whole curated dataset held on the training device as uint8, plus the fixed CV folds."""

    def __init__(self, device: torch.device, data_dir: str = None, manifest_path: str = None, cache_dir: str = None):
        self.manifest = load_manifest(manifest_path)
        kept = [m for m in self.manifest["images"] if m["status"] == "kept"]
        arr = build_cache(data_dir or C.DATA_DIR, self.manifest, cache_dir or C.OUT_DIR)
        self.device = device
        self.paths = [m["path"] for m in kept]
        self.images = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)
        self.labels = torch.tensor([m["label"] for m in kept], dtype=torch.long, device=device)
        self.folds = np.array([m["fold"] for m in kept])
        self.n_folds = self.manifest["n_folds"]
        self._mean = torch.tensor(C.NORM_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(C.NORM_STD, device=device).view(1, 3, 1, 1)

    def split(self, fold: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotation `fold`: test = fold, validation = next fold, training = the remaining folds."""
        val_fold = (fold + 1) % self.n_folds
        as_index = lambda mask: torch.from_numpy(np.flatnonzero(mask)).to(self.device)
        return (as_index((self.folds != fold) & (self.folds != val_fold)),
                as_index(self.folds == val_fold),
                as_index(self.folds == fold))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def eval_batch(self, idx: torch.Tensor) -> torch.Tensor:
        """Deterministic view: centre crop of the cached square (equivalent to Resize(256) + CenterCrop(224))."""
        o = (C.CACHE_SIZE - C.IMAGE_SIZE) // 2
        x = self.images[idx][:, :, o:o + C.IMAGE_SIZE, o:o + C.IMAGE_SIZE].float().div_(255.0)
        return self._normalize(x)

    def train_batch(self, idx: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        return self._normalize(augment(self.images[idx], gen))


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """
    Per-sample augmentation on the GPU, returning float images in [0, 1] at IMAGE_SIZE.

    Geometry (one affine resampling): random resized crop (area and aspect ratio), horizontal and
    vertical flip, rotation. Photometry: brightness, contrast, saturation (hue is left alone because
    lesion colour is diagnostic). Every image in the
    batch draws its own parameters from `gen`, so the stream is reproducible for a given seed.
    """
    B, dev = x.shape[0], x.device
    x = x.float().div_(255.0)

    def uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo

    area = uniform(*C.AUG_SCALE)
    ratio = torch.exp(uniform(math.log(C.AUG_RATIO[0]), math.log(C.AUG_RATIO[1])))
    w = torch.sqrt(area * ratio).clamp(max=1.0)      # crop width / height as fractions of the image
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = uniform(-1.0, 1.0) * (1.0 - w)              # crop centre in normalised [-1, 1] coordinates
    cy = uniform(-1.0, 1.0) * (1.0 - h)
    ang = uniform(-C.AUG_ROTATION_DEG, C.AUG_ROTATION_DEG) * (math.pi / 180.0)
    fx = torch.where(uniform(0.0, 1.0) < 0.5, -1.0, 1.0)
    fy = torch.where(uniform(0.0, 1.0) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)

    # output (u, v) -> flip -> rotate -> scale to the crop box -> translate to the crop centre
    theta = torch.empty(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * cos * fx
    theta[:, 0, 1] = -w * sin * fy
    theta[:, 0, 2] = cx
    theta[:, 1, 0] = h * sin * fx
    theta[:, 1, 1] = h * cos * fy
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, 3, C.IMAGE_SIZE, C.IMAGE_SIZE), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    gray_w = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
    x = x * uniform(1 - C.AUG_BRIGHTNESS, 1 + C.AUG_BRIGHTNESS).view(B, 1, 1, 1)
    mean = (x * gray_w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
    x = (x - mean) * uniform(1 - C.AUG_CONTRAST, 1 + C.AUG_CONTRAST).view(B, 1, 1, 1) + mean
    gray = (x * gray_w).sum(1, keepdim=True)
    x = (x - gray) * uniform(1 - C.AUG_SATURATION, 1 + C.AUG_SATURATION).view(B, 1, 1, 1) + gray

    return x.clamp_(0.0, 1.0)
