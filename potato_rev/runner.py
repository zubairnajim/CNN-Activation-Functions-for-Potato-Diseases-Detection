"""
potato_rev/runner.py
====================
Resumable, multi-process experiment runner.

The master builds the job list, then starts N workers that share one GPU. Each worker claims
the next unfinished job through an exclusive lock file, trains it, and writes one JSON per run
to <out>/runs/. A job whose JSON exists is done, so the command can be re-issued after any
interruption and only the missing runs are trained.

Usage
-----
  python -m potato_rev.runner --stages grid noaug scratch      # workers chosen from VRAM
  python -m potato_rev.runner --stages grid --workers 1
  python -m potato_rev.runner --stages full100                 # after the grid has finished
  python -m potato_rev.runner --stages grid --limit 4          # smoke test: first 4 jobs only
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import traceback

import torch

from potato_rev import config as C
from potato_rev.jobs import STAGES, build_jobs

MAX_ATTEMPTS = 3


def _paths(out_dir: str):
    dirs = {k: os.path.join(out_dir, k) for k in ("runs", "claims", "attempts", "logs", "weights")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _claim_next(jobs, dirs):
    for job in jobs:
        if os.path.exists(os.path.join(dirs["runs"], job["id"] + ".json")):
            continue
        try:
            os.close(os.open(os.path.join(dirs["claims"], job["id"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        return job
    return None


def worker(args) -> None:
    from potato_rev.data import PotatoData
    from potato_rev.engine import train_run

    dirs = _paths(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PotatoData(device, cache_dir=args.out)
    jobs = build_jobs(args.stages, args.out)[:args.limit]

    while True:
        job = _claim_next(jobs, dirs)
        if job is None:
            break
        claim = os.path.join(dirs["claims"], job["id"])
        attempts_file = os.path.join(dirs["attempts"], job["id"])
        with open(attempts_file, "a") as f:
            f.write("x")
        attempts = os.path.getsize(attempts_file)
        try:
            result = train_run(job, data, weights_dir=dirs["weights"])
            test = result.get("test") or {}
            print(f"[w{args.worker_id}] {job['id']}  {result['status']}  epochs={result['epochs_run']} "
                  f"best={result['best_epoch']}  val_loss={result['best_val_loss']}  "
                  f"test_acc={test.get('accuracy', float('nan')):.3f}  f1={test.get('f1_macro', float('nan')):.3f}  "
                  f"{result['train_time_s']:.0f}s", flush=True)
        except Exception as exc:
            oom = isinstance(exc, torch.OutOfMemoryError)
            print(f"[w{args.worker_id}] {job['id']}  ERROR (attempt {attempts}): {exc!r}"[:300], flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if oom:
                # a full GPU says nothing about the job: hand it back and wait for memory, without limit
                os.remove(attempts_file)
                os.remove(claim)
                time.sleep(60)
                continue
            if attempts < MAX_ATTEMPTS:
                time.sleep(2)
                os.remove(claim)
                continue
            result = dict(job, status="failed", error=traceback.format_exc()[-2000:])
        _write_json(os.path.join(dirs["runs"], job["id"] + ".json"), result)
        os.remove(claim)


def _auto_workers() -> int:
    if not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    # ~6.5 GB covers the largest job (VGG-19, batch 32, mixed precision) with headroom
    return max(1, min((os.cpu_count() or 2) - 1, int(vram_gb // 6.5)))


def master(args) -> None:
    from potato_rev.data import build_cache, load_manifest
    from potato_rev.sysinfo import system_specs

    dirs = _paths(args.out)
    specs = system_specs()
    _write_json(os.path.join(args.out, "system_specs.json"), specs)
    build_cache(C.DATA_DIR, load_manifest(), args.out)      # decode once, before the workers start
    for stale in glob.glob(os.path.join(dirs["claims"], "*")):
        os.remove(stale)
    for path in glob.glob(os.path.join(dirs["runs"], "*.json")):
        with open(path) as f:
            failed = json.load(f).get("status") == "failed"
        if failed:      # failed runs carry no result; clear them so they are trained again
            os.remove(path)
            attempts = os.path.join(dirs["attempts"], os.path.basename(path)[:-5])
            if os.path.exists(attempts):
                os.remove(attempts)

    jobs = build_jobs(args.stages, args.out)[:args.limit]
    done = lambda: sum(os.path.exists(os.path.join(dirs["runs"], j["id"] + ".json")) for j in jobs)
    start_done = done()
    n_workers = args.workers or _auto_workers()
    print(f"System: {specs['gpu']} ({specs['gpu_vram_gb']} GB) | {specs['cpu']} x{specs['cpu_logical_cores']} "
          f"| RAM {specs['ram_gb']} GB | torch {specs['pytorch']} | CUDA {specs['cuda']}")
    print(f"Stages: {args.stages} | jobs: {len(jobs)} | already done: {start_done} | workers: {n_workers}", flush=True)
    if start_done == len(jobs):
        return

    def spawn(i: int):
        log = open(os.path.join(dirs["logs"], f"worker_{i}.log"), "a")
        cmd = [sys.executable, "-m", "potato_rev.runner", "--worker", "--worker-id", str(i), "--out", args.out]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        cmd += ["--stages", *args.stages]
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)

    procs = {i: spawn(i) for i in range(n_workers)}
    respawns, t0, last_print = 0, time.time(), 0.0
    while True:
        time.sleep(5)
        n_done = done()
        alive = [i for i, p in procs.items() if p.poll() is None]
        if n_done >= len(jobs):
            break
        for i, p in procs.items():
            # a worker that died with jobs left (e.g. CUDA context failure) is restarted, within reason
            if p.poll() not in (None, 0) and respawns < 4 * n_workers:
                respawns += 1
                procs[i] = spawn(i)
        if not alive and all(p.poll() == 0 for p in procs.values()):
            break
        if time.time() - last_print >= args.report_every:
            last_print = time.time()
            elapsed = time.time() - t0
            rate = (n_done - start_done) / elapsed if elapsed > 0 else 0.0
            eta = (len(jobs) - n_done) / rate / 60 if rate > 0 else float("nan")
            print(f"[{elapsed / 60:6.1f} min] {n_done}/{len(jobs)} done | {rate * 60:.1f} runs/min "
                  f"| ETA {eta:.0f} min | workers alive {len(alive)} | respawns {respawns}", flush=True)
    for p in procs.values():
        p.wait()
    print(f"Finished: {done()}/{len(jobs)} jobs in {(time.time() - t0) / 60:.1f} min.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable multi-process runner for the revision experiments.")
    parser.add_argument("--stages", nargs="+", default=["grid"], choices=STAGES)
    parser.add_argument("--out", default=C.OUT_DIR)
    parser.add_argument("--workers", type=int, default=0, help="0 = choose from GPU memory")
    parser.add_argument("--limit", type=int, default=None, help="only the first N jobs (smoke test)")
    parser.add_argument("--report-every", type=float, default=60.0, help="seconds between progress lines")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        master(args)


if __name__ == "__main__":
    main()
