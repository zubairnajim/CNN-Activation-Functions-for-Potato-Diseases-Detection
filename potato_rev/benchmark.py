"""
potato_rev/benchmark.py
=======================
Computational cost of every architecture x activation pair: parameters, multiply-accumulate
operations, model size, single-image latency on GPU and CPU, and batched GPU throughput.

Usage
-----
  python -m potato_rev.benchmark                 # writes <out>/benchmark_<device>.json
  python -m potato_rev.benchmark --cpu-only
"""

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from potato_rev import config as C
from potato_rev.jobs import configurations
from potato_rev.models import build_model
from potato_rev.sysinfo import system_specs


def count_gmacs(model: nn.Module) -> float:
    """Multiply-accumulates (in billions) of the convolution and linear layers for one 224x224 image."""
    total = 0

    def conv_hook(m, _inp, out):
        nonlocal total
        total += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, _inp, out):
        nonlocal total
        total += out.shape[0] * m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook if isinstance(m, nn.Conv2d) else linear_hook)
             for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(1, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    return total / 1e9


def latency_ms(model: nn.Module, device: torch.device, batch: int, runs: int, warmup: int = 10) -> Dict[str, float]:
    """Wall-clock time per forward pass, synchronising the GPU so queued work is included."""
    x = torch.randn(batch, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for i in range(warmup + runs):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000.0)
    return {"mean": float(np.mean(times)), "std": float(np.std(times))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Computational benchmark of all configurations.")
    parser.add_argument("--out", default=C.OUT_DIR)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    cpu = torch.device("cpu")
    gpu = torch.device("cuda") if torch.cuda.is_available() and not args.cpu_only else None
    records = []
    for arch, act in sorted(configurations(), key=lambda p: (C.ARCHITECTURES.index(p[0]), p[1])):
        model, audit = build_model(arch, act, pretrained=False)
        model.eval()
        rec = {"arch": arch, "act": act, "native": audit["is_native"],
               "params_m": audit["params_millions"],
               "size_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20, 1),
               "gmacs": round(count_gmacs(model), 3)}
        c = latency_ms(model, cpu, batch=1, runs=20, warmup=3)
        rec.update({"cpu_ms": round(c["mean"], 2), "cpu_ms_std": round(c["std"], 2)})
        if gpu is not None:
            model.to(gpu)
            g1 = latency_ms(model, gpu, batch=1, runs=100)
            g32 = latency_ms(model, gpu, batch=32, runs=30)
            rec.update({"gpu_ms": round(g1["mean"], 2), "gpu_ms_std": round(g1["std"], 2),
                        "gpu_img_per_s_b32": round(32 * 1000.0 / g32["mean"], 1)})
        records.append(rec)
        print(rec, flush=True)
        del model
        if gpu is not None:
            torch.cuda.empty_cache()

    specs = system_specs()
    specs["cpu_threads"] = torch.get_num_threads()
    tag = "cpu" if gpu is None else "".join(ch if ch.isalnum() else "_" for ch in specs["gpu"])
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"benchmark_{tag}.json")
    with open(path, "w") as f:
        json.dump({"system": specs, "records": records}, f, indent=1)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
