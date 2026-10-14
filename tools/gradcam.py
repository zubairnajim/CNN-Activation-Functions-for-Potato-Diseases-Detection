"""
tools/gradcam.py
================
Grad-CAM visual explanations for one configuration.

The configuration is retrained on one CV rotation with the protocol of the main experiment
(the checkpoint of lowest validation loss is kept), then Grad-CAM heat maps are computed on
that rotation's test images: for each class, the first correctly and the first incorrectly
classified test image (where one exists).

Usage
-----
  python tools/gradcam.py --arch resnet50 --act lrelu --lr 1e-4 --fold 0
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from potato_rev import config as C  # noqa: E402
from potato_rev.data import PotatoData  # noqa: E402
from potato_rev.engine import train_run  # noqa: E402
from potato_rev.models import build_model  # noqa: E402


def target_layer(model: torch.nn.Module, arch: str) -> torch.nn.Module:
    """Last convolutional feature block of each backbone."""
    if arch.startswith("resnet"):
        return model.layer4
    if arch == "densenet121":
        return model.features
    if arch.startswith("vgg"):
        return model.features[-2]          # last activation before the final max-pool
    return model.features[-1]              # MobileNetV2 / EfficientNet: final 1x1 conv block


def gradcam(model, layer, x: torch.Tensor, cls: int) -> np.ndarray:
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    model.zero_grad(set_to_none=True)
    out = model(x)
    out[0, cls].backward()
    h1.remove()
    h2.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM for one configuration.")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--act", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", default=C.OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PotatoData(device, cache_dir=args.out)
    job = {"stage": "gradcam", "arch": args.arch, "act": args.act, "lr": args.lr, "fold": args.fold,
           "save_weights": True}
    job["id"] = f"gradcam__{args.arch}__{args.act}__lr{args.lr:.0e}__f{args.fold}"
    weights_dir = os.path.join(args.out, "weights")
    path = os.path.join(weights_dir, job["id"] + ".pt")
    if not os.path.isfile(path):
        result = train_run(job, data, weights_dir=weights_dir)
        print(f"retrained: test acc {result['test']['accuracy']:.3f}, macro-F1 {result['test']['f1_macro']:.3f}")

    model, _ = build_model(args.arch, args.act, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    # in-place activations would overwrite the tensors the backward hook needs
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    layer = target_layer(model, args.arch)

    _, _, test_idx = data.split(args.fold)
    with torch.no_grad():
        preds = model(data.eval_batch(test_idx)).argmax(1)
    labels = data.labels[test_idx]
    picks = []
    for c in range(C.NUM_CLASSES):
        for correct in (True, False):
            hit = ((labels == c) & ((preds == labels) == correct)).nonzero().flatten()
            if len(hit):
                picks.append((int(test_idx[hit[0]]), c, int(preds[hit[0]])))

    o = (C.CACHE_SIZE - C.IMAGE_SIZE) // 2
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.55, rows * 1.75))
    for ax in axes.ravel():
        ax.axis("off")
    for k, (i, true_c, pred_c) in enumerate(picks):
        img = data.images[i, :, o:o + C.IMAGE_SIZE, o:o + C.IMAGE_SIZE].permute(1, 2, 0).cpu().numpy()
        cam = gradcam(model, layer, data.eval_batch(torch.tensor([i], device=device)), pred_c)
        r, c0 = divmod(k, cols)
        axes[r, 2 * c0].imshow(img)
        axes[r, 2 * c0 + 1].imshow(img)
        axes[r, 2 * c0 + 1].imshow(cam, cmap="jet", alpha=0.45)
        ok = "correct" if true_c == pred_c else f"pred: {C.CLASS_NAMES[pred_c]}"
        axes[r, 2 * c0].set_title(f"{C.CLASS_NAMES[true_c]}", fontsize=6)
        axes[r, 2 * c0 + 1].set_title(ok, fontsize=6, color="green" if true_c == pred_c else "red")
    fig.suptitle(f"Grad-CAM: {args.arch} + {args.act} (rotation {args.fold + 1} test images)", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(args.out, "analysis", "figures"), exist_ok=True)
    save = os.path.join(args.out, "analysis", "figures", f"gradcam_{args.arch}_{args.act}.png")
    fig.savefig(save, dpi=300)
    print(f"[saved] {save}")


if __name__ == "__main__":
    main()
