"""
tools/audit_dataset.py
======================
Dataset integrity audit and cross-validation manifest builder.

The public dataset was compiled from web sources, so the same photograph can
appear more than once (sometimes under two different class labels). Leaving
those copies in place leaks test images into training and makes any reported
accuracy optimistic. This script:

  1. hashes every image (MD5 for byte-identical copies, a 256-bit difference
     hash for resized / re-compressed copies) and, as a second opinion, compares
     ImageNet-feature embeddings to surface candidates the hash can miss;
  2. groups duplicates and applies two curation rules
        - a group whose members carry *different* labels is excluded entirely
          (the ground truth is ambiguous),
        - a group within one class keeps a single representative;
  3. assigns the curated images to stratified folds, keeping photographs of the same
     specimen (RELATED_GROUPS) inside one fold;
  4. writes data_audit/manifest.json, the single source of truth consumed by
     the training pipeline (so every machine uses identical folds).

Usage
-----
  python tools/audit_dataset.py --report          # print candidates + montages only
  python tools/audit_dataset.py --write           # also write the manifest
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from potato_rev.config import CLASS_NAMES, N_FOLDS, SEED  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "potato_disease_dataset")
AUDIT_DIR = os.path.join(PROJECT_ROOT, "data_audit")

# A 256-bit dHash distance at or below this is treated as "same photograph".
DHASH_THRESHOLD = 20
# Pairs above the hash threshold that were confirmed by eye as the same photograph
# (re-cropped, re-scaled or colour-shifted), from the embedding-similarity montage (see --report).
MANUAL_DUPLICATE_PAIRS = [
    ("Black Scurf/20.jpg", "Common Scab/52.jpg"),
    ("Pink Rot/15.jpg", "Pink Rot/30.jpg"),
    ("Common Scab/15.jpg", "Common Scab/54.jpg"),
]
# Distinct photographs of the same specimen or the same photo series, also confirmed by eye.
# They are all kept, but forced into one fold so that no specimen straddles train and test.
RELATED_GROUPS = [
    ["Black Scurf/43.jpg", "Black Scurf/49.jpg", "Black Scurf/50.jpg"],
    ["Black Scurf/51.jpg", "Black Scurf/52.jpg", "Black Scurf/53.jpg",
     "Black Scurf/54.jpg", "Black Scurf/55.jpg", "Black Scurf/56.jpg"],
    ["Pink Rot/9.jpg", "Pink Rot/13.jpg"],
    ["Pink Rot/8.jpg", "Pink Rot/24.jpg"],
    ["Miscellaneous/6.jpg", "Miscellaneous/7.jpg"],
    ["Miscellaneous/22.jpg", "Miscellaneous/24.jpg"],
    ["Miscellaneous/21.jpg", "Miscellaneous/27.jpg"],
    ["Miscellaneous/71.jpg", "Miscellaneous/72.jpg"],
    ["Common Scab/44.jpg", "Common Scab/45.jpg"],
    ["Dry Rot/58.jpg", "Dry Rot/59.jpg"],
    ["Blackleg/18.jpeg", "Blackleg/19.jpg"],
]
# Embedding cosine similarity above which a pair is shown for manual review.
EMBED_REVIEW_THRESHOLD = 0.90


def natural_key(name: str):
    stem = os.path.splitext(name)[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def scan(root: str):
    records = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(root, cls)
        for fname in sorted(os.listdir(cls_dir), key=natural_key):
            path = os.path.join(cls_dir, fname)
            with open(path, "rb") as f:
                raw = f.read()
            im = Image.open(path)
            w, h = im.size
            gray = np.asarray(im.convert("RGB").convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
            records.append({
                "path": f"{cls}/{fname}",
                "label": CLASS_NAMES.index(cls),
                "md5": hashlib.md5(raw).hexdigest(),
                "width": w,
                "height": h,
                "mode": im.mode,
                "dhash": (gray[:, 1:] > gray[:, :-1]).flatten(),
            })
    return records


def embed(records, root: str) -> np.ndarray:
    """L2-normalised ImageNet ResNet-50 features, used only to surface review candidates."""
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(records), 32):
            batch = torch.stack([tf(Image.open(os.path.join(root, r["path"])).convert("RGB"))
                                 for r in records[i:i + 32]]).to(device)
            feats.append(torch.nn.functional.normalize(model(batch), dim=1).cpu().numpy())
    return np.concatenate(feats)


def union_find_groups(n: int, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def montage(records, pairs, root: str, save_path: str, tile: int = 190, per_row: int = 6):
    if not pairs:
        return

    def render(rec, note=""):
        im = Image.open(os.path.join(root, rec["path"])).convert("RGB")
        im.thumbnail((tile, tile))
        canvas = Image.new("RGB", (tile, tile + 28), "white")
        canvas.paste(im, ((tile - im.width) // 2, 28 + (tile - im.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.text((2, 1), rec["path"][:32], fill="black")
        d.text((2, 13), note, fill=(180, 0, 0))
        return canvas

    rows = (len(pairs) + per_row - 1) // per_row
    sheet = Image.new("RGB", (per_row * (tile + 6), rows * 2 * (tile + 34)), (210, 210, 210))
    for k, (i, j, note) in enumerate(pairs):
        x = (k % per_row) * (tile + 6)
        y = (k // per_row) * 2 * (tile + 34)
        sheet.paste(render(records[i], note), (x, y))
        sheet.paste(render(records[j]), (x, y + tile + 30))
    sheet.save(save_path)


def stratified_group_folds(labels: np.ndarray, units: list, n_folds: int, seed: int) -> np.ndarray:
    """
    Stratified k-fold over *units* (a unit is one image, or a set of related images of one class
    that must stay together). Within each class the units are shuffled, then dealt largest-first
    to whichever fold currently holds the fewest images of that class.
    """
    rng = np.random.RandomState(seed)
    folds = np.full(len(labels), -1, dtype=int)
    total = np.zeros(n_folds, dtype=int)
    for c in np.unique(labels):
        class_units = [u for u in units if labels[u[0]] == c]
        order = rng.permutation(len(class_units))
        class_units = sorted((class_units[i] for i in order), key=len, reverse=True)
        count = np.zeros(n_folds, dtype=int)
        for unit in class_units:
            # fewest of this class first, then fewest overall, to keep fold sizes even
            k = min(range(n_folds), key=lambda f: (count[f], total[f]))
            folds[unit] = k
            count[k] += len(unit)
            total[k] += len(unit)
    return folds


def main():
    parser = argparse.ArgumentParser(description="Audit the potato dataset and build the CV manifest.")
    parser.add_argument("--report", action="store_true", help="Print duplicate candidates and save review montages.")
    parser.add_argument("--write", action="store_true", help="Write data_audit/manifest.json.")
    parser.add_argument("--no-embed", action="store_true", help="Skip the embedding-similarity review pass.")
    args = parser.parse_args()
    os.makedirs(AUDIT_DIR, exist_ok=True)

    records = scan(RAW_DATA_DIR)
    n = len(records)
    index_of = {r["path"]: i for i, r in enumerate(records)}
    print(f"Scanned {n} images in {len(CLASS_NAMES)} classes.")

    D = np.stack([r["dhash"] for r in records])
    ham = (D[:, None, :] != D[None, :, :]).sum(-1)
    iu, ju = np.triu_indices(n, 1)

    hash_pairs = [(int(i), int(j)) for i, j in zip(iu, ju) if ham[i, j] <= DHASH_THRESHOLD]
    md5_pairs = [(int(i), int(j)) for i, j in zip(iu, ju) if records[i]["md5"] == records[j]["md5"]]
    manual_pairs = [(index_of[a], index_of[b]) for a, b in MANUAL_DUPLICATE_PAIRS]
    dup_pairs = sorted(set(hash_pairs) | set(md5_pairs) | set(manual_pairs))
    print(f"Byte-identical pairs: {len(md5_pairs)} | dHash<={DHASH_THRESHOLD} pairs: {len(hash_pairs)} "
          f"| manually confirmed: {len(manual_pairs)}")

    if args.report:
        montage(records, [(i, j, f"dhash={ham[i, j]}") for i, j in dup_pairs], RAW_DATA_DIR,
                os.path.join(AUDIT_DIR, "review_hash_duplicates.png"))
        borderline = [(int(i), int(j), f"dhash={ham[i, j]}") for i, j in zip(iu, ju)
                      if DHASH_THRESHOLD < ham[i, j] <= 60]
        borderline.sort(key=lambda t: ham[t[0], t[1]])
        print(f"Borderline hash pairs ({DHASH_THRESHOLD} < d <= 60): {len(borderline)}")
        montage(records, borderline[:36], RAW_DATA_DIR, os.path.join(AUDIT_DIR, "review_hash_borderline.png"))
        if not args.no_embed:
            E = embed(records, RAW_DATA_DIR)
            cos = E @ E.T
            known = set(dup_pairs)
            cand = [(int(i), int(j), f"cos={cos[i, j]:.3f} d={ham[i, j]}") for i, j in zip(iu, ju)
                    if cos[i, j] >= EMBED_REVIEW_THRESHOLD and (int(i), int(j)) not in known]
            cand.sort(key=lambda t: -cos[t[0], t[1]])
            print(f"Embedding candidates (cos >= {EMBED_REVIEW_THRESHOLD}, not already hash-matched): {len(cand)}")
            for i, j, note in cand[:60]:
                print(f"   {records[i]['path']:28s} {records[j]['path']:28s} {note}")
            montage(records, cand[:48], RAW_DATA_DIR, os.path.join(AUDIT_DIR, "review_embedding_candidates.png"))

    # ---- curation ----------------------------------------------------------
    groups = union_find_groups(n, dup_pairs)
    status = ["kept"] * n
    group_of = [None] * n
    group_report = []
    for gid, members in enumerate(groups):
        labels = {records[i]["label"] for i in members}
        conflict = len(labels) > 1
        for i in members:
            group_of[i] = gid
        if conflict:
            for i in members:
                status[i] = "excluded_label_conflict"
        else:
            keep = max(members, key=lambda i: (records[i]["width"] * records[i]["height"], -i))
            for i in members:
                if i != keep:
                    status[i] = "excluded_duplicate"
        group_report.append({
            "group": gid,
            "label_conflict": conflict,
            "members": [records[i]["path"] for i in members],
            "kept": [records[i]["path"] for i in members if status[i] == "kept"],
        })

    kept = [i for i in range(n) if status[i] == "kept"]
    labels = np.array([records[i]["label"] for i in kept])

    # related images that survived curation travel as one unit
    pos = {i: k for k, i in enumerate(kept)}
    related_of = [None] * n
    units, in_unit = [], set()
    for rid, paths in enumerate(RELATED_GROUPS):
        members = [index_of[p] for p in paths if status[index_of[p]] == "kept"]
        assert len({records[i]["label"] for i in members}) == 1, f"related group {rid} spans classes"
        for i in members:
            related_of[i] = rid
        units.append([pos[i] for i in members])
        in_unit.update(members)
    units += [[pos[i]] for i in kept if i not in in_unit]
    folds = stratified_group_folds(labels, units, N_FOLDS, SEED)

    print(f"\nDuplicate groups: {len(groups)} "
          f"({sum(g['label_conflict'] for g in group_report)} with conflicting labels)")
    for g in group_report:
        tag = "CONFLICT" if g["label_conflict"] else "dup"
        print(f"   [{tag}] {g['members']} -> kept {g['kept']}")
    print(f"\nCurated dataset: {len(kept)} images "
          f"(removed {status.count('excluded_duplicate')} redundant copies, "
          f"{status.count('excluded_label_conflict')} label-conflicting images)")
    header = f"{'Class':20s} {'raw':>5s} {'curated':>8s} | " + " ".join(f"f{k}" .rjust(4) for k in range(N_FOLDS))
    print(header)
    for c, cls in enumerate(CLASS_NAMES):
        raw_c = sum(r["label"] == c for r in records)
        per_fold = [int(((labels == c) & (folds == k)).sum()) for k in range(N_FOLDS)]
        print(f"{cls:20s} {raw_c:5d} {int((labels == c).sum()):8d} | " + " ".join(f"{v:4d}" for v in per_fold))
    print(f"{'Total':20s} {n:5d} {len(kept):8d} | " +
          " ".join(f"{int((folds == k).sum()):4d}" for k in range(N_FOLDS)))

    if args.write:
        fold_of = {i: int(f) for i, f in zip(kept, folds)}
        manifest = {
            "description": "Curated potato tuber disease dataset and fixed stratified CV folds.",
            "class_names": CLASS_NAMES,
            "n_folds": N_FOLDS,
            "seed": SEED,
            "dhash_threshold": DHASH_THRESHOLD,
            "n_raw": n,
            "n_curated": len(kept),
            "duplicate_groups": group_report,
            "related_groups": RELATED_GROUPS,
            "images": [{
                "path": r["path"],
                "label": r["label"],
                "md5": r["md5"],
                "width": r["width"],
                "height": r["height"],
                "status": status[i],
                "related_group": related_of[i],
                "fold": fold_of.get(i, -1),
            } for i, r in enumerate(records)],
        }
        out = os.path.join(AUDIT_DIR, "manifest.json")
        with open(out, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
