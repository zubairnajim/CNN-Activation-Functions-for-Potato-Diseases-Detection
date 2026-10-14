import math
import os
import random
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from potato_rev import config as C
from potato_rev.data import PotatoData
from potato_rev.metrics import classification_metrics
from potato_rev.models import build_model


def _evaluate(model: nn.Module, data: PotatoData, idx: torch.Tensor, criterion: nn.Module,
              amp: bool) -> Tuple[float, torch.Tensor]:
    """Mean loss and float32 logits over `idx` in evaluation mode."""
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(idx), C.EVAL_BATCH_SIZE):
            x = data.eval_batch(idx[i:i + C.EVAL_BATCH_SIZE]).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=amp):
                logits.append(model(x).float())
    logits = torch.cat(logits)
    return float(criterion(logits, data.labels[idx]).item()), logits


def _summarise(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    out = classification_metrics(y_true, prob.argmax(1), prob)
    out["nll"] = float(nn.functional.cross_entropy(logits, labels).item())
    out["y_true"] = y_true.tolist()
    out["y_prob"] = np.round(prob, 4).tolist()
    return out


def train_run(job: Dict[str, Any], data: PotatoData, weights_dir: str = None) -> Dict[str, Any]:
    """
    Fine-tune one (architecture, activation, learning rate) configuration on one CV rotation.

    The validation fold drives LR scheduling, early stopping and checkpoint selection; the test
    fold is evaluated exactly once, with the checkpoint of lowest validation loss.
    """
    device = data.device
    amp = C.USE_AMP and device.type == "cuda"
    fold, lr = job["fold"], job["lr"]
    max_epochs = job.get("max_epochs", C.MAX_EPOCHS)
    early_stop = job.get("early_stop", True)
    use_aug = job.get("augment", True)

    # One seed per rotation, shared by every configuration: identical head initialisation,
    # mini-batch order and augmentation stream, so configurations are compared pairwise.
    seed = C.SEED + 1000 * fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    train_idx, val_idx, test_idx = data.split(fold)
    model, audit = build_model(job["arch"], job["act"], pretrained=job.get("pretrained", True))
    model.to(device, memory_format=torch.channels_last)

    # No weight decay on biases, normalisation parameters and PReLU slopes.
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": C.WEIGHT_DECAY},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=C.LR_PLATEAU_FACTOR, patience=C.LR_PLATEAU_PATIENCE, min_lr=C.MIN_LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    warmup_steps = C.WARMUP_EPOCHS * math.ceil(len(train_idx) / C.BATCH_SIZE)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "grad_norm")}
    best_val_loss, best_epoch, best_state, wait, step = float("inf"), 0, None, 0, 0
    status = "ok"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=gen, device=device)]
        loss_sum, correct, seen, norms = 0.0, 0, 0, []
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            if len(idx) < 2:      # batch norm cannot train on a single sample
                continue
            x = data.train_batch(idx, gen) if use_aug else data.eval_batch(idx)
            x = x.contiguous(memory_format=torch.channels_last)
            y = data.labels[idx]
            if step < warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = lr * (step + 1) / warmup_steps
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = criterion(out.float(), y)
            if not torch.isfinite(loss):
                status = "diverged"
                break
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            loss_sum += float(loss.item()) * len(idx)
            correct += int((out.argmax(1) == y).sum().item())
            seen += len(idx)
            if math.isfinite(norm):
                norms.append(norm)
        if status != "ok":
            break

        val_loss, val_logits = _evaluate(model, data, val_idx, criterion, amp)
        current_lr = optimizer.param_groups[0]["lr"]
        if epoch > C.WARMUP_EPOCHS:
            scheduler.step(val_loss)

        history["train_loss"].append(round(loss_sum / max(seen, 1), 4))
        history["train_acc"].append(round(correct / max(seen, 1), 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(float((val_logits.argmax(1) == data.labels[val_idx]).float().mean().item()), 4))
        history["lr"].append(current_lr)
        history["grad_norm"].append(round(float(np.mean(norms)), 4) if norms else None)

        if not math.isfinite(val_loss):
            status = "diverged"
            break
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            wait += 1
            if early_stop and wait >= C.EARLY_STOPPING_PATIENCE:
                break

    train_time = time.time() - t0
    result = dict(job)
    result.update({
        "status": status,
        "seed": seed,
        "audit": audit,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "epochs_run": len(history["val_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "train_time_s": round(train_time, 2),
        "epoch_time_s": round(train_time / max(len(history["val_loss"]), 1), 3),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated(device) / 2 ** 20, 1) if device.type == "cuda" else None,
        "history": history,
    })

    if best_state is not None:
        model.load_state_dict(best_state)
        result["val"] = _summarise(_evaluate(model, data, val_idx, criterion, amp)[1], data.labels[val_idx])
        result["test"] = _summarise(_evaluate(model, data, test_idx, criterion, amp)[1], data.labels[test_idx])
        if job.get("save_weights") and weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            torch.save(best_state, os.path.join(weights_dir, f"{job['id']}.pt"))
    else:
        result["status"] = "diverged"
    return result
