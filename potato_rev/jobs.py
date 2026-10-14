import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

from potato_rev import config as C

SCRATCH_LR_GRID = (1e-4, 3e-4, 1e-3)   # randomly initialised networks need larger steps
STAGES = ("grid", "noaug", "scratch", "full100")


def _job(stage: str, arch: str, act: str, lr: float, fold: int, **extra) -> Dict[str, Any]:
    job = {"stage": stage, "arch": arch, "act": act, "lr": lr, "fold": fold}
    job.update(extra)
    job["id"] = f"{stage}__{arch}__{act}__lr{lr:.0e}__f{fold}"
    return job


def configurations() -> List[tuple]:
    """All architecture x activation pairs, plus the native activation where it is not one of the six."""
    pairs = [(arch, act) for act in C.ACTIVATIONS for arch in C.ARCHITECTURES]
    pairs += [(arch, nat) for arch, nat in C.NATIVE_ACTIVATION.items() if nat not in C.ACTIVATIONS]
    return pairs


def load_results(out_dir: str, stage: str) -> List[Dict[str, Any]]:
    results = []
    for path in sorted(glob.glob(os.path.join(out_dir, "runs", f"{stage}__*.json"))):
        with open(path, "r") as f:
            results.append(json.load(f))
    return results


def modal_selected_lr(out_dir: str) -> Dict[str, float]:
    """For each architecture with its native activation: the LR most often chosen by validation loss."""
    by_key = defaultdict(list)
    for r in load_results(out_dir, "grid"):
        if r["act"] == C.NATIVE_ACTIVATION[r["arch"]] and r.get("best_val_loss") is not None:
            by_key[(r["arch"], r["fold"])].append((r["best_val_loss"], r["lr"]))
    votes = defaultdict(Counter)
    for (arch, _fold), runs in by_key.items():
        votes[arch][min(runs)[1]] += 1
    return {arch: c.most_common(1)[0][0] for arch, c in votes.items()}


def build_jobs(stages: List[str], out_dir: str) -> List[Dict[str, Any]]:
    """
    Job list in execution order. Architectures vary fastest so that concurrent workers hold
    different backbones (the two VGGs are the memory-hungry ones).
    """
    jobs = []
    folds = range(C.N_FOLDS)
    if "grid" in stages:
        jobs += [_job("grid", arch, act, lr, fold)
                 for fold in folds for lr in C.LR_GRID for arch, act in configurations()]
        jobs += [_job("grid", arch, act, C.LR_GRID_EXTENSION[arch], fold)
                 for fold in folds for arch, act in configurations()]
    if "noaug" in stages:
        jobs += [_job("noaug", arch, C.NATIVE_ACTIVATION[arch], lr, fold, augment=False)
                 for fold in folds for arch in C.ARCHITECTURES for lr in C.lr_grid(arch)]
    if "scratch" in stages:
        jobs += [_job("scratch", arch, C.NATIVE_ACTIVATION[arch], lr, fold, pretrained=False)
                 for fold in folds for lr in SCRATCH_LR_GRID for arch in C.ARCHITECTURES]
    if "full100" in stages:
        # full 100-epoch curves without early stopping, at the LR the grid selected most often
        chosen = modal_selected_lr(out_dir)
        jobs += [_job("full100", arch, C.NATIVE_ACTIVATION[arch], chosen[arch], 0, early_stop=False)
                 for arch in C.ARCHITECTURES if arch in chosen]
    return jobs
