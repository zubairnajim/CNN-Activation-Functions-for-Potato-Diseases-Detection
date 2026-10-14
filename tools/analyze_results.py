"""
tools/analyze_results.py
========================
Turns the per-run JSON files written by potato_rev.runner into the manuscript's tables,
statistical tests and figures. Safe to run on partial results.

Protocol recap: for every (architecture, activation, CV rotation) three learning rates are
trained; the run with the lowest *validation* loss is selected and only that run's test-fold
predictions are reported. Test data never influences a choice.

Outputs (under <out>/analysis/)
  tables/   CSV files, one per manuscript table
  figures/  PNG files (300 dpi)
  summary.txt  headline numbers and test results, for writing the text

Usage
-----
  python tools/analyze_results.py                       # reads revision_outputs/
  python tools/analyze_results.py --out some/other/dir
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from potato_rev import config as C  # noqa: E402
from potato_rev.metrics import binary_auc, classification_metrics  # noqa: E402

ARCH_LABEL = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3", "vgg16": "VGG-16", "vgg19": "VGG-19",
}
ACT_LABEL = {"relu": "ReLU", "relu6": "ReLU6", "lrelu": "Leaky ReLU", "prelu": "PReLU",
             "mish": "Mish", "swish": "Swish", "gelu": "GELU"}
METRICS = ["accuracy", "error_rate", "precision_macro", "recall_macro", "specificity_macro", "f1_macro", "auc_macro"]
CHANCE = 1.0 / C.NUM_CLASSES

plt.rcParams.update({"figure.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Loading and learning-rate selection
# ---------------------------------------------------------------------------

def load_runs(out_dir: str) -> list:
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "runs", "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def select_by_validation(runs: list, stage: str) -> pd.DataFrame:
    """One row per (arch, act, fold): the LR candidate with the lowest validation loss."""
    groups = {}
    for r in runs:
        if r["stage"] == stage:
            groups.setdefault((r["arch"], r["act"], r["fold"]), []).append(r)
    rows = []
    for (arch, act, fold), cands in sorted(groups.items()):
        usable = [r for r in cands if r.get("test") and r.get("best_val_loss") is not None]
        row = {"arch": arch, "act": act, "fold": fold, "n_candidates": len(cands),
               "n_diverged": sum(r["status"] != "ok" for r in cands)}
        if usable:
            best = min(usable, key=lambda r: r["best_val_loss"])
            row.update({m: best["test"][m] for m in METRICS})
            row.update({"lr": best["lr"], "best_epoch": best["best_epoch"], "epochs_run": best["epochs_run"],
                        "val_loss": best["best_val_loss"], "val_accuracy": best["val"]["accuracy"],
                        "train_time_s": best["train_time_s"], "epoch_time_s": best["epoch_time_s"],
                        "peak_mem_mb": best.get("peak_mem_mb"), "run": best})
        else:
            # every candidate diverged: the configuration failed on this rotation
            row.update({m: np.nan for m in METRICS})
            row.update({"lr": np.nan, "run": None})
        rows.append(row)
    return pd.DataFrame(rows)


def pooled(sel: pd.DataFrame, arch: str, act: str):
    """Out-of-fold test predictions of one configuration, concatenated over the rotations."""
    y_true, y_prob = [], []
    for run in sel[(sel.arch == arch) & (sel.act == act)]["run"]:
        if run is not None:
            y_true += run["test"]["y_true"]
            y_prob += run["test"]["y_prob"]
    return np.array(y_true), np.array(y_prob)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, fn, n_boot: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = [fn(y_true[i], y_pred[i]) for i in (rng.randint(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def summarise(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, act), g in sel.groupby(["arch", "act"], sort=False):
        row = {"arch": arch, "act": act, "native": act == C.NATIVE_ACTIVATION[arch],
               "folds": int(g["accuracy"].notna().sum()), "failed_folds": int(g["accuracy"].isna().sum())}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1)
        y_true, y_prob = pooled(sel, arch, act)
        if len(y_true):
            y_pred = y_prob.argmax(1)
            pm = classification_metrics(y_true, y_pred, y_prob)
            row["pooled_accuracy"], row["pooled_f1_macro"] = pm["accuracy"], pm["f1_macro"]
            row["pooled_n"] = len(y_true)
            row["acc_ci_lo"], row["acc_ci_hi"] = bootstrap_ci(y_true, y_pred, lambda a, b: float((a == b).mean()))
        lrs = g["lr"].dropna()
        row["lr_mode"] = lrs.mode().iloc[0] if len(lrs) else np.nan
        row["lr_counts"] = "; ".join(f"{lr:.0e}x{n}" for lr, n in lrs.value_counts().sort_index().items())
        row["best_epoch_mean"] = g["best_epoch"].mean() if "best_epoch" in g else np.nan
        row["epochs_run_mean"] = g["epochs_run"].mean() if "epochs_run" in g else np.nan
        row["train_time_s_mean"] = g["train_time_s"].mean() if "train_time_s" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Architecture x activation table of 'mean +/- std' (percent); the native activation is starred."""
    acts = C.ACTIVATIONS + ["relu6"]
    table = pd.DataFrame(index=[ARCH_LABEL[a] for a in C.ARCHITECTURES], columns=[ACT_LABEL[a] for a in acts])
    for _, r in summary.iterrows():
        cell = f"{100 * r[f'{metric}_mean']:.1f} ± {100 * r[f'{metric}_std']:.1f}" + ("*" if r["native"] else "")
        table.loc[ARCH_LABEL[r["arch"]], ACT_LABEL[r["act"]]] = cell
    return table.dropna(axis=1, how="all").fillna("-")


def per_class_table(sel: pd.DataFrame, configs: list) -> pd.DataFrame:
    rows = []
    for arch, act in configs:
        y_true, y_prob = pooled(sel, arch, act)
        if not len(y_true):
            continue
        for pc in classification_metrics(y_true, y_prob.argmax(1), y_prob)["per_class"]:
            rows.append({"model": f"{ARCH_LABEL[arch]} + {ACT_LABEL[act]}", **pc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def holm(pvals: list) -> list:
    order = np.argsort(pvals)
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def friedman_block(wide: pd.DataFrame, label: str, lines: list) -> pd.DataFrame:
    """Friedman test over the columns of `wide` (rows are blocks), then Wilcoxon signed-rank with Holm."""
    wide = wide.dropna()
    if wide.shape[0] < 3 or wide.shape[1] < 3:
        lines.append(f"{label}: not enough complete blocks ({wide.shape[0]}).")
        return pd.DataFrame()
    chi2, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean()
    kendall_w = chi2 / (wide.shape[0] * (wide.shape[1] - 1))
    lines.append(f"{label}: Friedman chi2={chi2:.2f}, df={wide.shape[1] - 1}, p={p:.4g}, "
                 f"Kendall W={kendall_w:.3f}, blocks={wide.shape[0]}")
    lines.append("   mean ranks (1 = best): " + ", ".join(f"{c}={r:.2f}" for c, r in ranks.sort_values().items()))
    pairs, raw = list(combinations(wide.columns, 2)), []
    for a, b in pairs:
        diff = wide[a] - wide[b]
        raw.append(1.0 if np.allclose(diff, 0) else stats.wilcoxon(wide[a], wide[b]).pvalue)
    adj = holm(raw)
    out = pd.DataFrame([{"a": a, "b": b, "mean_diff_pp": 100 * (wide[a] - wide[b]).mean(),
                         "p_wilcoxon": p0, "p_holm": p1} for (a, b), p0, p1 in zip(pairs, raw, adj)])
    sig = out[out.p_holm < 0.05]
    lines.append(f"   pairwise Wilcoxon-Holm: {len(sig)}/{len(out)} pairs significant at 0.05" +
                 ("" if sig.empty else ": " + "; ".join(f"{r.a}>{r.b}" if r.mean_diff_pp > 0 else f"{r.b}>{r.a}"
                                                        for r in sig.itertuples())))
    return out


def run_statistics(sel: pd.DataFrame, tables_dir: str, lines: list) -> None:
    six = sel[sel.act.isin(C.ACTIVATIONS)].copy()
    # a configuration that failed on a rotation is scored at chance level rather than dropped
    six["f1_filled"] = six["f1_macro"].fillna(CHANCE)

    lines.append("\n== Effect of the activation function (blocks = architecture x rotation) ==")
    wide = six.pivot_table(index=["arch", "fold"], columns="act", values="f1_filled")[C.ACTIVATIONS]
    friedman_block(wide.rename(columns=ACT_LABEL), "All architectures", lines).to_csv(
        os.path.join(tables_dir, "stats_activation_pairwise.csv"), index=False)
    for arch in C.ARCHITECTURES:
        if arch in wide.index.get_level_values(0):
            friedman_block(wide.loc[arch].rename(columns=ACT_LABEL), f"  within {ARCH_LABEL[arch]}", lines)

    lines.append("\n== Effect of the architecture (blocks = activation x rotation) ==")
    wide = six.pivot_table(index=["act", "fold"], columns="arch", values="f1_filled")[C.ARCHITECTURES]
    friedman_block(wide.rename(columns=ARCH_LABEL), "All activations", lines).to_csv(
        os.path.join(tables_dir, "stats_architecture_pairwise.csv"), index=False)

    lines.append("\n== Swaps within versus across activation families (paired over architecture x rotation) ==")
    family = {"relu": "piecewise-linear", "relu6": "piecewise-linear", "lrelu": "piecewise-linear",
              "prelu": "piecewise-linear", "swish": "smooth", "mish": "smooth", "gelu": "smooth"}
    rows = []
    for (arch, fold), g in sel.groupby(["arch", "fold"]):
        nat = C.NATIVE_ACTIVATION[arch]
        base = g[g.act == nat]["f1_macro"]
        if base.empty:
            continue
        for r in g[g.act.isin(C.ACTIVATIONS) & (g.act != nat)].itertuples():
            rows.append({"arch": arch, "fold": fold, "cross": family[r.act] != family[nat],
                         "delta": (r.f1_macro if pd.notna(r.f1_macro) else CHANCE) - base.iloc[0]})
    fam = pd.DataFrame(rows)
    fam_rows = []
    for group, archs in (("batch-normalised", [a for a in C.ARCHITECTURES if not a.startswith("vgg")]),
                         ("BN-free (VGG)", [a for a in C.ARCHITECTURES if a.startswith("vgg")])):
        blocks = fam[fam.arch.isin(archs)].groupby(["arch", "fold", "cross"])["delta"].mean().unstack()
        if blocks.shape[1] < 2 or len(blocks) < 3:
            continue
        p_cross = stats.wilcoxon(blocks[True]).pvalue
        p_within = stats.wilcoxon(blocks[False]).pvalue
        p_diff = stats.wilcoxon(blocks[True], blocks[False]).pvalue
        fam_rows.append({"networks": group, "blocks": len(blocks),
                         "cross_family_delta_pp": 100 * blocks[True].mean(), "cross_sd_pp": 100 * blocks[True].std(ddof=1),
                         "within_family_delta_pp": 100 * blocks[False].mean(), "within_sd_pp": 100 * blocks[False].std(ddof=1),
                         "p_cross_vs_0": p_cross, "p_within_vs_0": p_within, "p_cross_vs_within": p_diff})
        lines.append(f"   {group}: cross-family {100 * blocks[True].mean():+.2f} pp (p vs 0 = {p_cross:.2g}), "
                     f"within-family {100 * blocks[False].mean():+.2f} pp (p = {p_within:.2g}); "
                     f"cross vs within p = {p_diff:.2g} over {len(blocks)} blocks")
    pd.DataFrame(fam_rows).to_csv(os.path.join(tables_dir, "stats_activation_family.csv"), index=False)

    lines.append("\n== Swapped activation versus the native one (paired over rotations, macro-F1) ==")
    rows = []
    for arch in C.ARCHITECTURES:
        nat = C.NATIVE_ACTIVATION[arch]
        base = sel[(sel.arch == arch) & (sel.act == nat)].set_index("fold")["f1_macro"]
        for act in C.ACTIVATIONS:
            if act == nat:
                continue
            other = sel[(sel.arch == arch) & (sel.act == act)].set_index("fold")["f1_macro"].fillna(CHANCE)
            both = pd.concat([base, other], axis=1, keys=["native", "swap"]).dropna()
            if len(both) < 3:
                continue
            diff = both["swap"] - both["native"]
            p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(both["swap"], both["native"]).pvalue
            rows.append({"arch": ARCH_LABEL[arch], "native": ACT_LABEL[nat], "swap": ACT_LABEL[act],
                         "delta_f1_pp": 100 * diff.mean(), "delta_sd_pp": 100 * diff.std(ddof=1),
                         "wins": int((diff > 0).sum()), "folds": len(both), "p_wilcoxon": p})
    vs = pd.DataFrame(rows)
    if not vs.empty:
        vs["p_holm"] = holm(vs["p_wilcoxon"].tolist())
        vs.to_csv(os.path.join(tables_dir, "stats_swap_vs_native.csv"), index=False)
        lines.append(f"   {int((vs.delta_f1_pp > 0).sum())}/{len(vs)} swaps have a higher mean macro-F1 than native; "
                     f"{int((vs.p_holm < 0.05).sum())} significant after Holm correction "
                     f"(5 rotations: the smallest attainable two-sided Wilcoxon p is 0.0625).")
        lines.append(f"   mean delta over all swaps: {vs.delta_f1_pp.mean():+.2f} pp "
                     f"(range {vs.delta_f1_pp.min():+.2f} to {vs.delta_f1_pp.max():+.2f})")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_heatmap(summary: pd.DataFrame, metric: str, title: str, path: str) -> None:
    acts = [a for a in C.ACTIVATIONS + ["relu6"] if a in set(summary.act)]
    mean = np.full((len(C.ARCHITECTURES), len(acts)), np.nan)
    text = [[""] * len(acts) for _ in C.ARCHITECTURES]
    for _, r in summary.iterrows():
        i, j = C.ARCHITECTURES.index(r["arch"]), acts.index(r["act"])
        mean[i, j] = 100 * r[f"{metric}_mean"]
        text[i][j] = f"{mean[i, j]:.1f}\n±{100 * r[f'{metric}_std']:.1f}" + ("*" if r["native"] else "")
    fig, ax = plt.subplots(figsize=(1.15 * len(acts) + 1.6, 0.52 * len(C.ARCHITECTURES) + 1.2))
    im = ax.imshow(mean, cmap="YlGnBu", aspect="auto")
    lo, hi = np.nanmin(mean), np.nanmax(mean)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if text[i][j]:
                ax.text(j, i, text[i][j], ha="center", va="center", fontsize=7,
                        color="white" if (mean[i, j] - lo) > 0.6 * (hi - lo) else "black")
    ax.set_xticks(range(len(acts)), [ACT_LABEL[a] for a in acts])
    ax.set_yticks(range(len(C.ARCHITECTURES)), [ARCH_LABEL[a] for a in C.ARCHITECTURES])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="%", fraction=0.035, pad=0.02)
    fig.savefig(path)
    plt.close(fig)


def fig_boxplots(sel: pd.DataFrame, path: str) -> None:
    six = sel[sel.act.isin(C.ACTIVATIONS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, key, order, labels, name in ((axes[0], "arch", C.ARCHITECTURES, ARCH_LABEL, "architecture"),
                                         (axes[1], "act", C.ACTIVATIONS, ACT_LABEL, "activation function")):
        data = [100 * six[six[key] == k]["f1_macro"].dropna().values for k in order]
        ax.boxplot(data, widths=0.6, showmeans=True, medianprops={"color": "black"},
                   meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4})
        ax.set_xticks(range(1, len(order) + 1), [labels[k] for k in order], rotation=30, ha="right")
        ax.set_title(f"By {name}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Test macro-F1 (%)")
    fig.savefig(path)
    plt.close(fig)


def fig_learning_curves(runs: list, title: str, path: str) -> None:
    """Loss and accuracy curves, one panel pair per architecture."""
    by_arch = {r["arch"]: r for r in runs}
    archs = [a for a in C.ARCHITECTURES if a in by_arch]
    if not archs:
        return
    fig, axes = plt.subplots(2, len(archs), figsize=(2.3 * len(archs), 4.4), sharex=True, squeeze=False)
    for j, arch in enumerate(archs):
        r, h = by_arch[arch], by_arch[arch]["history"]
        x = range(1, len(h["val_loss"]) + 1)
        axes[0, j].plot(x, h["train_loss"], label="train")
        axes[0, j].plot(x, h["val_loss"], label="validation")
        axes[1, j].plot(x, [100 * v for v in h["train_acc"]])
        axes[1, j].plot(x, [100 * v for v in h["val_acc"]])
        for ax in axes[:, j]:
            ax.axvline(r["best_epoch"], color="grey", ls="--", lw=0.8)
            ax.grid(alpha=0.3)
        axes[0, j].set_title(f"{ARCH_LABEL[arch]}\n{ACT_LABEL[r['act']]}, LR {r['lr']:.0e}", fontsize=8)
        axes[1, j].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, y=1.02, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(sel: pd.DataFrame, configs: list, path: str) -> None:
    short = ["B.Scurf", "Blackleg", "C.Scab", "Dry Rot", "Healthy", "Misc.", "Pink Rot"]
    fig, axes = plt.subplots(1, len(configs), figsize=(4.1 * len(configs), 3.9), squeeze=False)
    for ax, (arch, act) in zip(axes[0], configs):
        y_true, y_prob = pooled(sel, arch, act)
        cm = np.array(classification_metrics(y_true, y_prob.argmax(1))["confusion_matrix"])
        frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(short)):
            for j in range(len(short)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(short)), short, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_title(f"{ARCH_LABEL[arch]} + {ACT_LABEL[act]}\n(pooled over 5 test folds, n={cm.sum()})", fontsize=9)
    axes[0, 0].set_ylabel("True")
    fig.savefig(path)
    plt.close(fig)


def fig_roc(sel: pd.DataFrame, configs: list, path: str) -> None:
    fig, axes = plt.subplots(1, len(configs), figsize=(4.0 * len(configs), 3.8), squeeze=False)
    for ax, (arch, act) in zip(axes[0], configs):
        y_true, y_prob = pooled(sel, arch, act)
        for c, name in enumerate(C.CLASS_NAMES):
            pos, score = y_true == c, y_prob[:, c]
            thr = np.r_[np.inf, np.sort(np.unique(score))[::-1]]
            tpr = [(score[pos] >= t).mean() for t in thr]
            fpr = [(score[~pos] >= t).mean() for t in thr]
            ax.plot(fpr, tpr, lw=1.1, label=f"{name} ({binary_auc(pos, score):.3f})")
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_title(f"{ARCH_LABEL[arch]} + {ACT_LABEL[act]}", fontsize=9)
        ax.legend(fontsize=6, title="Class (AUC)", title_fontsize=6, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("True positive rate")
    fig.savefig(path)
    plt.close(fig)


def fig_swap_deltas(tables_dir: str, path: str) -> None:
    """Paired change in macro-F1 when a backbone's native activation is replaced, by activation family."""
    path_csv = os.path.join(tables_dir, "stats_swap_vs_native.csv")
    if not os.path.exists(path_csv):
        return
    v = pd.read_csv(path_csv)
    smooth = {"Swish", "Mish", "GELU"}
    v["cross"] = v.apply(lambda r: (r["native"] in smooth) != (r["swap"] in smooth), axis=1)
    order = [ARCH_LABEL[a] for a in C.ARCHITECTURES]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhline(0, color="black", lw=0.8)
    markers = {"ReLU": "o", "Leaky ReLU": "s", "PReLU": "^", "Swish": "D", "Mish": "v", "GELU": "P"}
    for cross, color, label in ((False, "#2a7ab9", "same family as native"), (True, "#d9534f", "other family than native")):
        s = v[v.cross == cross]
        for act, m in markers.items():
            t = s[s.swap == act]
            x = [order.index(a) + (list(markers).index(act) - 2.5) * 0.09 for a in t.arch]
            ax.errorbar(x, t.delta_f1_pp, yerr=t.delta_sd_pp / np.sqrt(t.folds), fmt=m, color=color, ms=5,
                        elinewidth=0.7, capsize=0, label=f"{act}" if not cross else None)
    ax.set_xticks(range(len(order)), order, rotation=25, ha="right")
    ax.set_ylabel("Δ macro-F1 vs native (pp)")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#2a7ab9", marker="o", ls="", label="same family as native"),
               Line2D([], [], color="#d9534f", marker="o", ls="", label="other family than native")]
    handles += [Line2D([], [], color="grey", marker=m, ls="", label=a) for a, m in markers.items()]
    ax.legend(handles=handles, fontsize=7, ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.set_title("Effect of replacing the native activation (mean ± SE over 5 rotations)")
    fig.savefig(path)
    plt.close(fig)


def fig_lr_sensitivity(runs: list, path: str) -> None:
    """Mean test macro-F1 at each candidate LR, per architecture; divergent runs count as chance."""
    grid = pd.DataFrame([{"arch": r["arch"], "act": r["act"], "lr": r["lr"],
                          "f1": r["test"]["f1_macro"] if r.get("test") and r["status"] == "ok" else CHANCE}
                         for r in runs if r["stage"] == "grid"])
    if grid.empty:
        return
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.8), sharey=True)
    for ax, arch in zip(axes.ravel(), C.ARCHITECTURES):
        g = grid[grid.arch == arch]
        for act in [a for a in C.ACTIVATIONS + ["relu6"] if a in set(g.act)]:
            m = g[g.act == act].groupby("lr")["f1"].mean()
            ax.plot(range(len(m)), 100 * m.values, marker="o", ms=3, lw=1,
                    ls="-" if act == C.NATIVE_ACTIVATION[arch] else "--", label=ACT_LABEL[act])
            ax.set_xticks(range(len(m)), [f"{lr:.0e}" for lr in m.index])
        ax.set_title(ARCH_LABEL[arch], fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test macro-F1 (%)")
    for ax in axes[1]:
        ax.set_xlabel("Learning rate")
    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, title="solid = native")
    fig.subplots_adjust(right=0.9, hspace=0.38)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate revision runs into tables, tests and figures.")
    parser.add_argument("--out", default=C.OUT_DIR)
    args = parser.parse_args()

    tables_dir = os.path.join(args.out, "analysis", "tables")
    figures_dir = os.path.join(args.out, "analysis", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    runs = load_runs(args.out)
    lines = [f"Runs loaded: {len(runs)} | by stage: {pd.Series([r['stage'] for r in runs]).value_counts().to_dict()}",
             f"Run status: {pd.Series([r['status'] for r in runs]).value_counts().to_dict()}"]

    sel = select_by_validation(runs, "grid")
    summary = summarise(sel).sort_values("f1_macro_mean", ascending=False)
    summary.drop(columns=[]).to_csv(os.path.join(tables_dir, "all_configurations.csv"), index=False)
    for metric in ("f1_macro", "accuracy"):
        matrix_table(summary, metric).to_csv(os.path.join(tables_dir, f"matrix_{metric}.csv"))
    sel.drop(columns=["run"]).to_csv(os.path.join(tables_dir, "per_fold_selected.csv"), index=False)

    complete = summary[summary.folds == C.N_FOLDS]
    lines.append(f"\nConfigurations: {len(summary)} ({len(complete)} with all {C.N_FOLDS} rotations)")
    lines.append("\n== Top 10 by mean macro-F1 (mean ± SD over rotations; pooled accuracy with 95% bootstrap CI) ==")
    for r in summary.head(10).itertuples():
        lines.append(f"  {ARCH_LABEL[r.arch]:16s} {ACT_LABEL[r.act]:10s} F1 {100 * r.f1_macro_mean:5.1f} ± {100 * r.f1_macro_std:4.1f} "
                     f"| acc {100 * r.accuracy_mean:5.1f} ± {100 * r.accuracy_std:4.1f} "
                     f"| pooled acc {100 * r.pooled_accuracy:5.1f} [{100 * r.acc_ci_lo:.1f}, {100 * r.acc_ci_hi:.1f}] "
                     f"| AUC {r.auc_macro_mean:.3f} | LR {r.lr_counts}")
    lines.append("\n== Bottom 5 ==")
    for r in summary.tail(5).itertuples():
        lines.append(f"  {ARCH_LABEL[r.arch]:16s} {ACT_LABEL[r.act]:10s} F1 {100 * r.f1_macro_mean:5.1f} ± {100 * r.f1_macro_std:4.1f} "
                     f"| failed folds {r.failed_folds} | LR {r.lr_counts}")

    grid_runs = [r for r in runs if r["stage"] == "grid"]
    div = pd.DataFrame([{"arch": r["arch"], "act": r["act"], "lr": r["lr"], "bad": r["status"] != "ok"} for r in grid_runs])
    if not div.empty:
        lines.append(f"\n== Divergence: {int(div.bad.sum())}/{len(div)} grid runs ==")
        d = div[div.bad].groupby(["arch", "act", "lr"]).size()
        lines += [f"  {ARCH_LABEL[a]} + {ACT_LABEL[b]} at LR {lr:.0e}: {n} rotation(s)" for (a, b, lr), n in d.items()]
        chosen = sel.dropna(subset=["lr"])
        low = (chosen.lr == chosen.arch.map(lambda a: min(C.lr_grid(a)))).mean()
        high = (chosen.lr == chosen.arch.map(lambda a: max(C.lr_grid(a)))).mean()
        lines.append(f"\n== Selected LR: {chosen.lr.value_counts().sort_index().to_dict()} "
                     f"(lowest value of the architecture's grid chosen in {100 * low:.0f}% of cases, "
                     f"highest in {100 * high:.0f}%) ==")
        pd.crosstab(chosen.arch, chosen.lr).to_csv(os.path.join(tables_dir, "selected_lr_counts.csv"))

    run_statistics(sel, tables_dir, lines)

    top = [(r.arch, r.act) for r in complete.head(3).itertuples()]
    if top:
        per_class_table(sel, top).to_csv(os.path.join(tables_dir, "per_class_top3.csv"), index=False)
        fig_confusion(sel, top, os.path.join(figures_dir, "confusion_top3.png"))
        fig_roc(sel, top, os.path.join(figures_dir, "roc_top3.png"))
        lines.append("\n== Per-class results of the best configuration (pooled) ==")
        best = per_class_table(sel, top[:1])
        lines += [f"  {r['class']:18s} P {100 * r['precision']:5.1f}  R {100 * r['recall']:5.1f}  "
                  f"F1 {100 * r['f1']:5.1f}  AUC {r['auc']:.3f}  n={r['support']}" for _, r in best.iterrows()]

    fig_heatmap(summary, "f1_macro", "Test macro-F1 (%), mean ± SD over 5 rotations (* native activation)",
                os.path.join(figures_dir, "heatmap_f1.png"))
    fig_heatmap(summary, "accuracy", "Test accuracy (%), mean ± SD over 5 rotations (* native activation)",
                os.path.join(figures_dir, "heatmap_accuracy.png"))
    if not sel.empty:
        fig_boxplots(sel, os.path.join(figures_dir, "boxplots_f1.png"))
    fig_lr_sensitivity(runs, os.path.join(figures_dir, "lr_sensitivity.png"))
    fig_swap_deltas(tables_dir, os.path.join(figures_dir, "swap_deltas.png"))

    # learning curves: the selected native-activation run of rotation 0, and the 100-epoch runs
    native0 = [row["run"] for _, row in sel.iterrows()
               if row["fold"] == 0 and row["act"] == C.NATIVE_ACTIVATION[row["arch"]] and row["run"] is not None]
    fig_learning_curves(native0, "Early-stopped training (rotation 1); dashed line = selected checkpoint",
                        os.path.join(figures_dir, "learning_curves_early_stopped.png"))
    fig_learning_curves([r for r in runs if r["stage"] == "full100"],
                        "Full 100 epochs without early stopping (rotation 1); dashed line = lowest validation loss",
                        os.path.join(figures_dir, "learning_curves_100_epochs.png"))

    # ablations: augmentation and transfer learning, native activation
    base = summary[summary.native].set_index("arch")
    ablation_rows = []
    for stage, label in (("noaug", "no augmentation"), ("scratch", "no ImageNet pre-training")):
        s = select_by_validation(runs, stage)
        if s.empty:
            continue
        for arch, g in s.groupby("arch", sort=False):
            if arch not in base.index:
                continue
            ref = sel[(sel.arch == arch) & (sel.act == C.NATIVE_ACTIVATION[arch])].set_index("fold")["f1_macro"]
            abl = g.set_index("fold")["f1_macro"].fillna(CHANCE)
            both = pd.concat([ref, abl], axis=1, keys=["full", "ablated"]).dropna()
            ablation_rows.append({
                "ablation": label, "arch": ARCH_LABEL[arch], "folds": len(both),
                "full_f1": 100 * both["full"].mean(), "ablated_f1": 100 * both["ablated"].mean(),
                "ablated_f1_sd": 100 * both["ablated"].std(ddof=1),
                "delta_pp": 100 * (both["ablated"] - both["full"]).mean(),
                "ablated_accuracy": 100 * g["accuracy"].fillna(CHANCE).mean(),
                "epochs_run": g["epochs_run"].mean() if "epochs_run" in g else np.nan})
    if ablation_rows:
        abl = pd.DataFrame(ablation_rows)
        abl.to_csv(os.path.join(tables_dir, "ablations.csv"), index=False)
        lines.append("\n== Ablations (native activation, macro-F1 %, paired over rotations) ==")
        for name, g in abl.groupby("ablation"):
            lines.append(f"  {name}: mean delta {g.delta_pp.mean():+.1f} pp over {len(g)} architectures")
            lines += [f"     {r.arch:16s} full {r.full_f1:5.1f} -> ablated {r.ablated_f1:5.1f} ({r.delta_pp:+.1f}) [{r.folds} folds]"
                      for r in g.itertuples()]

    # training cost per architecture (selected runs)
    cost = sel.dropna(subset=["lr"]).groupby("arch").agg(
        epochs_run=("epochs_run", "mean"), best_epoch=("best_epoch", "mean"),
        epoch_time_s=("epoch_time_s", "mean"), train_time_s=("train_time_s", "mean"),
        peak_mem_mb=("peak_mem_mb", "max")).reindex(C.ARCHITECTURES).rename(index=ARCH_LABEL)
    cost.round(2).to_csv(os.path.join(tables_dir, "training_cost.csv"))

    with open(os.path.join(args.out, "analysis", "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
