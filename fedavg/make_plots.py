"""Figures of the report, written to results/figures/. Run after the sweeps:

    python -m fedavg.make_plots            # all figures
    python -m fedavg.make_plots --only data main
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch

from . import core, data
from .sweep import NOISE_SCALES

RES = "results"
FIG = os.path.join(RES, "figures")
MAJORITY = 0.0577  # share of the most frequent class (label 24) in the training data
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})


def hist(path):
    return pd.read_csv(os.path.join(path, "history.csv"))


def summary(path):
    return json.load(open(os.path.join(path, "summary.json")))


def stack(paths, col):
    return np.array([hist(p)[col].values for p in paths])


def band(ax, x, ys, label, color=None, ls="-"):
    m = ys.mean(0)
    line, = ax.plot(x, m, label=label, color=color, ls=ls, lw=1.3)
    if len(ys) > 1:
        ax.fill_between(x, ys.min(0), ys.max(0), color=line.get_color(), alpha=0.18, lw=0)
    return line


def plain_log_axis(ax):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="minor", labelsize=7)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(FIG, name))


# ------------------------------------------------------------------ data
def fig_data():
    clients, (xt, yt) = data.load_raw()
    K = len(clients)
    counts = np.array([np.bincount(y, minlength=62) for _, y in clients])
    n = counts.sum(1)
    total = counts.sum(0)
    p = total / total.sum()
    tv = 0.5 * np.abs(counts / n[:, None] - p).sum(1)
    rng = np.random.default_rng(0)
    tv_iid = np.array([[0.5 * np.abs(rng.multinomial(nk, p) / nk - p).sum() for nk in n] for _ in range(200)])
    present = (counts > 0).sum(1)
    order = np.argsort(tv)
    # five different clients, each an extreme of one property
    chosen = {}
    for name, ranking in [("fewest images", np.argsort(n)), ("most images", np.argsort(-n)),
                          ("fewest classes", np.argsort(present)), ("closest to pooled mix", order),
                          ("largest single-class share", np.argsort(-(counts.max(1) / n)))]:
        chosen[name] = int(next(k for k in ranking if int(k) not in chosen.values()))
    test_counts = np.bincount(yt, minlength=62)
    facts = dict(n_clients=K, n_total=int(n.sum()), n_min=int(n.min()), n_max=int(n.max()), n_median=float(np.median(n)),
                 majority_class=data.CLASS_NAMES[int(total.argmax())], majority_share=float(p.max()),
                 rarest_class=data.CLASS_NAMES[int(total.argmin())], rarest_count=int(total.min()),
                 share_digits=float(p[:10].sum()), share_upper=float(p[10:36].sum()), share_lower=float(p[36:].sum()),
                 classes_present_min=int(present.min()), classes_present_median=float(np.median(present)), classes_present_max=int(present.max()),
                 tv_mean=float(tv.mean()), tv_min=float(tv.min()), tv_max=float(tv.max()), tv_iid_mean=float(tv_iid.mean()),
                 tv_iid_max=float(tv_iid.max()), tv_train_test=float(0.5 * np.abs(p - test_counts / test_counts.sum()).sum()),
                 imbalance_ratio=float(total.max() / total.min()),
                 chosen={k: dict(client=v, n=int(n[v]), classes=int(present[v]), tv=float(tv[v]),
                                 top=[(data.CLASS_NAMES[i], int(counts[v, i])) for i in np.argsort(-counts[v])[:3]]) for k, v in chosen.items()})
    os.makedirs(FIG, exist_ok=True)
    json.dump(facts, open(os.path.join(FIG, "data_facts.json"), "w"), indent=1)

    colors = ["#4c72b0"] * 10 + ["#dd8452"] * 26 + ["#55a868"] * 26
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.25, 1, 0.05], hspace=0.7, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    ax.bar(range(62), 100 * p, color=colors)
    ax.plot(range(62), 100 * test_counts / test_counts.sum(), "k_", ms=7, label="test data")
    ax.axhline(100 / 62, color="0.4", ls="--", lw=0.8, label="uniform, 1.6%")
    ax.set_xticks(range(62)); ax.set_xticklabels(data.CLASS_NAMES, fontsize=7)
    ax.set_ylabel("share of images (%)"); ax.set_xlim(-0.7, 61.7)
    ax.set_title(f"All {K} clients, {n.sum():,} images: digits (blue) {p[:10].sum():.0%}, upper case (orange) {p[10:36].sum():.0%}, lower case (green) {p[36:].sum():.0%}")
    ax.legend(loc="upper right")
    for i, (name, k) in enumerate(chosen.items()):
        ax = fig.add_subplot(gs[1, i])
        ax.bar(range(62), counts[k], color=colors, width=0.9)
        ax.set_title(f"client {k}: {name}\n{n[k]} images, {present[k]} classes\nTV distance {tv[k]:.2f}", fontsize=7.5)
        ax.set_xticks([0, 10, 36, 61]); ax.set_xticklabels(["0", "A", "a", "z"])
        if i == 0:
            ax.set_ylabel("images")
    save(fig, "data_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.9, 1]))
    share = counts[order] / n[order][:, None]
    im = axes[0].imshow(share, aspect="auto", cmap="magma", vmin=0, vmax=0.12, interpolation="nearest")
    axes[0].set_xticks(range(62)); axes[0].set_xticklabels(data.CLASS_NAMES, fontsize=6)
    axes[0].set_ylabel("client (sorted by distance from the pooled distribution)")
    axes[0].set_title("Share of each class inside each client (black = class absent)")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label="share of the client's images", extend="max")
    axes[1].hist(tv_iid.ravel(), bins=40, density=True, color="0.6", label="random deal of the same sizes (200 draws)")
    axes[1].hist(tv, bins=25, density=True, color="#c44e52", alpha=0.8, label="the 100 real clients")
    axes[1].set_xlabel("total-variation distance, client vs pooled class distribution"); axes[1].set_ylabel("density")
    axes[1].set_title(f"Real mean {tv.mean():.3f}; random-deal mean {tv_iid.mean():.3f}")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "client_class_heatmap.png")


# ------------------------------------------------------------------ Part 1.1
def fig_main():
    paths = sorted(glob.glob(os.path.join(RES, "main", "seed*")))
    x = hist(paths[0])["round"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    band(axes[0], x, stack(paths, "train_loss"), "training (all 100 clients)")
    band(axes[0], x, stack(paths, "val_loss"), "validation (all 100 clients)")
    axes[0].axhline(np.log(62), color="0.4", ls="--", lw=0.8, label="ln 62 (uniform prediction)")
    axes[0].set_ylabel("cross-entropy loss"); plain_log_axis(axes[0])
    band(axes[1], x, stack(paths, "train_acc"), "training (all 100 clients)")
    band(axes[1], x, stack(paths, "val_acc"), "validation (all 100 clients)")
    band(axes[1], x[1:], stack(paths, "sel_val_acc")[:, 1:], "validation, the round's 10 clients only", color="0.55")
    axes[1].axhline(MAJORITY, color="0.4", ls="--", lw=0.8, label="always predict the most frequent class")
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("communication round"); ax.legend(fontsize=7)
    c = summary(paths[0])["config"]
    fig.suptitle(f"FedAvg, C = 10% (10 clients), E = {c['epochs']}, B = {c['batch_size']}, learning rate {c['lr']}: "
                 f"mean and min-max range of {len(paths)} seeds", fontsize=10)
    fig.tight_layout()
    save(fig, "main_curves.png")


def fig_tune():
    rows = []
    for p in glob.glob(os.path.join(RES, "tune", "*")):
        m = re.match(r"lr([\d.]+)_B(\d+)", os.path.basename(p))
        h = hist(p)
        rows.append(dict(lr=float(m.group(1)), B=int(m.group(2)), val=h.val_acc[-20:].mean(), train=h.train_acc[-20:].mean(),
                         vloss=h.val_loss.iloc[-1]))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(FIG, "tune_table.csv"), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title in zip(axes, ["val", "train", "vloss"], ["validation accuracy, rounds 281-300", "training accuracy, rounds 281-300", "validation loss, round 300"]):
        t = d.pivot(index="B", columns="lr", values=col)
        im = ax.imshow(t.values, aspect="auto", cmap="viridis" if col != "vloss" else "viridis_r",
                       vmin=0.6 if col != "vloss" else None, vmax=None if col != "vloss" else 1.4)
        ax.set_xticks(range(len(t.columns))); ax.set_xticklabels(t.columns)
        ax.set_yticks(range(len(t.index))); ax.set_yticklabels(t.index)
        ax.set_xlabel("client learning rate"); ax.set_ylabel("batch size B"); ax.set_title(title); ax.grid(False)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                ax.text(j, i, f"{t.values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if (t.values[i, j] < 0.75) == (col != "vloss") else "black")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Learning rate and batch size (C = 10%, E = 5, seed 0, 300 rounds)", fontsize=10)
    fig.tight_layout()
    save(fig, "tune_heatmap.png")


def ce_table():
    rows = []
    for p in glob.glob(os.path.join(RES, "ce", "*")):
        m = re.match(r"C([\d.]+)_E(\d+)_seed(\d+)", os.path.basename(p))
        h = hist(p)
        hit = np.flatnonzero(h.val_acc.values >= 0.75)
        rows.append(dict(C=float(m.group(1)), E=int(m.group(2)), seed=int(m.group(3)), path=p,
                         train=h.train_acc[-20:].mean(), val=h.val_acc[-20:].mean(), val_loss=h.val_loss.iloc[-1],
                         best_val=h.val_acc.max(), val_sd=h.val_acc[-50:].std(), test=summary(p)["test_acc"],
                         r75=int(hit[0]) if len(hit) else np.nan,
                         s75=h.samples_processed[hit[0]] if len(hit) else np.nan))
    return pd.DataFrame(rows)


def fig_ce():
    d = ce_table()
    d.drop(columns="path").to_csv(os.path.join(FIG, "ce_table.csv"), index=False)
    Cs, Es = sorted(d.C.unique()), sorted(d.E.unique())
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(2, len(Cs), figsize=(3.3 * len(Cs), 6), sharex=True, sharey=True)
    for j, C in enumerate(Cs):
        for i, E in enumerate(Es):
            paths = list(d[(d.C == C) & (d.E == E)].path)
            x = hist(paths[0])["round"].values
            col = cmap(i / (len(Es) - 0.5))
            band(axes[0, j], x, stack(paths, "train_acc"), f"E = {E}", color=col)
            band(axes[1, j], x, stack(paths, "val_acc"), f"E = {E}", color=col)
        axes[0, j].set_title(f"C = {C:.0%} ({int(round(C * 100))} client{'s' if C > 0.01 else ''} per round)")
        axes[1, j].set_xlabel("communication round")
    axes[0, 0].set_ylabel("training accuracy"); axes[1, 0].set_ylabel("validation accuracy")
    axes[0, 0].set_ylim(0, 1); axes[1, -1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Client fraction C and local epochs E: mean and min-max range of 3 seeds", fontsize=10)
    fig.tight_layout()
    save(fig, "ce_curves.png")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    g = d.groupby(["C", "E"]).mean(numeric_only=True).reset_index()
    for ax, col, title in zip(axes, ["train", "val", "val_sd", "r75"],
                              ["training accuracy, rounds 281-300", "validation accuracy, rounds 281-300",
                               "std of validation accuracy over rounds 251-300", "rounds to 75% validation accuracy"]):
        t = g.pivot(index="C", columns="E", values=col)
        im = ax.imshow(t.values, aspect="auto", cmap="viridis" if col in ("train", "val") else "viridis_r")
        ax.set_xticks(range(len(t.columns))); ax.set_xticklabels(t.columns)
        ax.set_yticks(range(len(t.index))); ax.set_yticklabels([f"{c:.0%}" for c in t.index])
        ax.set_xlabel("local epochs E"); ax.set_ylabel("client fraction C"); ax.set_title(title, fontsize=9); ax.grid(False)
        fmt = "{:.3f}" if col != "r75" else "{:.0f}"
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, "never" if np.isnan(v) else fmt.format(v), ha="center", va="center", fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.35, lw=0, pad=0.8))
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save(fig, "ce_final.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, E in enumerate(Es):
        for C, ls in zip([0.01, 0.1], [":", "-"]):
            paths = list(d[(d.C == C) & (d.E == E)].path)
            if not paths:
                continue
            x = hist(paths[0])["samples_processed"].values[1:]
            ax.plot(x, stack(paths, "val_acc").mean(0)[1:], ls=ls, color=cmap(i / (len(Es) - 0.5)), label=f"E = {E}, C = {C:.0%}")
    ax.set_xscale("log"); ax.set_xlabel("training samples processed by all clients (samples x epochs)")
    ax.set_ylabel("validation accuracy"); ax.set_ylim(0.3, 0.85); ax.legend(fontsize=6, ncol=2)
    ax.set_title("The same runs against client computation", fontsize=10)
    fig.tight_layout()
    save(fig, "ce_compute.png")


def fig_clients():
    """Non-standard plot: validation accuracy of every client in every round, with its selections."""
    a = np.load(os.path.join(RES, "main", "seed0", "arrays.npz"))
    acc, sel = a["client_val_acc"], a["selected"]
    order = np.argsort(acc[-1])
    R = acc.shape[0] - 1
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.5, 1.5, 1]))
    for ax, hi in zip(axes[:2], [40, R]):
        im = ax.imshow(acc[:hi + 1, order].T, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower", interpolation="nearest")
        if hi <= 60:
            rr, cc = np.nonzero(sel[:hi + 1, order])
            ax.plot(rr, cc, ".", color="white", ms=2.5)
        ax.set_xlabel("communication round"); ax.grid(False)
        ax.set_title(f"rounds 0-{hi}" + (" (white dot: client selected in that round)" if hi <= 60 else ""), fontsize=9)
    axes[0].set_ylabel("client, sorted by final validation accuracy")
    fig.colorbar(im, ax=axes[1], label="validation accuracy of the global model on the client's own data")
    d = np.diff(acc, axis=0)
    s = sel[1:]
    windows = [(1, 10), (11, 50), (51, 150), (151, R)]
    xs = np.arange(len(windows))
    gain_sel = [d[lo - 1:hi][s[lo - 1:hi]].mean() for lo, hi in windows]
    gain_not = [d[lo - 1:hi][~s[lo - 1:hi]].mean() for lo, hi in windows]
    axes[2].bar(xs - 0.2, gain_sel, 0.4, label="clients selected in the round")
    axes[2].bar(xs + 0.2, gain_not, 0.4, label="clients not selected")
    axes[2].set_xticks(xs); axes[2].set_xticklabels([f"{lo}-{hi}" for lo, hi in windows])
    axes[2].set_xlabel("rounds"); axes[2].set_ylabel("mean one-round change of the client's validation accuracy")
    axes[2].legend(fontsize=7); axes[2].set_title("Does being selected help the client itself?", fontsize=9)
    json.dump(dict(windows=windows, gain_selected=gain_sel, gain_not_selected=gain_not,
                   final_min=float(acc[-1].min()), final_max=float(acc[-1].max()), final_std=float(acc[-1].std()),
                   final_p10=float(np.percentile(acc[-1], 10)), final_p90=float(np.percentile(acc[-1], 90))),
              open(os.path.join(FIG, "client_heatmap.json"), "w"), indent=1, default=float)
    fig.tight_layout()
    save(fig, "client_heatmap.png")


# ------------------------------------------------------------------ Part 1.2
def fig_ray():
    paths = sorted(glob.glob(os.path.join(RES, "ray", "seed*")))
    x = hist(paths[0])["round"].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    band(axes[0], x, stack(paths, "train_loss"), "training"); band(axes[0], x, stack(paths, "val_loss"), "validation")
    axes[0].set_ylabel("cross-entropy loss"); plain_log_axis(axes[0])
    band(axes[1], x, stack(paths, "train_acc"), "training"); band(axes[1], x, stack(paths, "val_acc"), "validation")
    axes[1].axhline(MAJORITY, color="0.4", ls="--", lw=0.8, label="most frequent class")
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("communication round"); ax.legend(fontsize=7)
    c = summary(paths[0])["config"]
    fig.suptitle(f"FedAvg with Ray actors, 4 clients per round, E = {c['epochs']}, B = {c['batch_size']}, learning rate {c['lr']}: "
                 f"mean and min-max range of {len(paths)} seeds (all 100 clients evaluated)", fontsize=10)
    fig.tight_layout()
    save(fig, "ray_curves.png")

    t = np.load(os.path.join(paths[0], "arrays.npz"))["timings"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    t0 = t[t[:, 0] == 1][:, 5].min()
    cols = plt.get_cmap("tab10")
    for rnd in range(1, 9):
        r = t[t[:, 0] == rnd]
        for slot, row in enumerate(r[np.argsort(r[:, 1])]):
            ax.barh(slot, row[4] - row[5], left=row[5] - t0, height=0.8, color=cols(rnd % 10))
            ax.text(row[4] - t0, slot, f" {int(row[1])}", va="center", fontsize=7)
        ax.axvline(r[:, 5].min() - t0, color="0.3", lw=0.6, ls=":")
        ax.text(r[:, 5].min() - t0, 3.62, f" round {rnd}", fontsize=7, color="0.3")
    ax.set_ylim(-0.6, 3.95)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"actor {i + 1}" for i in range(4)])
    ax.set_xlabel("seconds since the first actor of round 1 finished its constructor")
    ax.set_title("Rounds 1-8 of the Ray run (seed 0): each bar is one actor, from the end of its constructor to its returned update; "
                 "numbers are client ids", fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    save(fig, "ray_gantt.png")


# ------------------------------------------------------------------ Part 2
def noise_paths(b):
    return sorted(glob.glob(os.path.join(RES, "noise", f"b{b}_seed*")))


def fig_noise():
    scales = [b for b in NOISE_SCALES if noise_paths(b)]
    show = [b for b in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0] if b in scales]
    cmap = plt.get_cmap("viridis_r")
    colour = {b: cmap(0.1 + 0.9 * i / (len(show) - 1)) for i, b in enumerate(show)}
    sel = noise_lr_selected() if glob.glob(os.path.join(RES, "noise_lr", "*")) else None
    nrow = 2 if sel is not None else 1
    fig, axes = plt.subplots(nrow, 3, figsize=(14, 3.9 * nrow), sharey=True, squeeze=False)
    for b in show:
        paths = noise_paths(b)
        x = hist(paths[0])["round"].values
        band(axes[0, 0], x, stack(paths, "train_acc"), f"b = {b}", color=colour[b])
        band(axes[0, 1], x, stack(paths, "val_acc"), f"b = {b}", color=colour[b])
        band(axes[0, 2], x, stack(paths, "clean_train_acc" if b > 0 else "train_acc"), f"b = {b}", color=colour[b])
    if sel is not None:
        h0 = hist(noise_paths(0.0)[0])
        for j, c in enumerate(["train_acc", "val_acc", "train_acc"]):
            axes[1, j].plot(h0["round"], h0[c], color=colour[0.0], lw=1.2, label="b = 0.0, rate 0.5")
        for b, r in sel.iterrows():
            h = hist(r["path"])
            for j, c in enumerate(["train_acc", "val_acc", "clean_train_acc"]):
                axes[1, j].plot(h["round"], h[c], color=colour.get(b, "k"), lw=1.2, label=f"b = {b}, rate {r['lr']}")
    titles = ["perturbed training images", "perturbed validation images", "the same training images without noise"]
    rows = ["accuracy, learning rate 0.5 (3 seeds)", "accuracy, learning rate chosen per b (seed 0)"]
    for i, row in enumerate(axes):
        for ax, title in zip(row, titles):
            ax.set_title(title, fontsize=9); ax.set_xlabel("communication round")
            ax.axhline(MAJORITY, color="0.4", ls="--", lw=0.8)
        row[0].set_ylabel(rows[i], fontsize=9); row[0].set_ylim(0, 1)
        row[2].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("FedAvg on Laplace-perturbed images, C = 10%, E = 5, B = 20 (top: mean and min-max range of 3 seeds; "
                 "dashed line: most frequent class)", fontsize=10)
    fig.tight_layout()
    save(fig, "noise_curves.png")

    rows = []
    for b in scales:
        ps = noise_paths(b)
        last = lambda c: np.array([hist(p)[c].iloc[-1] for p in ps])
        rows.append(dict(b=b, train=last("train_acc"), val=last("val_acc"), clean=last("clean_train_acc") if b > 0 else last("train_acc"),
                         test=np.array([summary(p)["test_acc"] for p in ps])))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bs = [r["b"] for r in rows]
    pos = np.arange(len(bs))                      # one equally spaced position per tested scale
    where = {b: i for i, b in enumerate(bs)}
    for key, lab, mk in [("train", "training accuracy (perturbed images), learning rate 0.5", "o"),
                         ("test", "test accuracy (clean held-out images), learning rate 0.5", "s"),
                         ("val", "validation accuracy (perturbed images), learning rate 0.5", "^"),
                         ("clean", "training images without noise, learning rate 0.5", "x")]:
        m = np.array([r[key].mean() for r in rows]); lo = np.array([r[key].min() for r in rows]); hi = np.array([r[key].max() for r in rows])
        ax.errorbar(pos, m, yerr=[m - lo, hi - m], marker=mk, ms=4, lw=1.2, capsize=2, label=lab,
                    ls="-" if key in ("train", "test") else ":")
    if sel is not None:
        sp = [where[b] for b in sel.index]
        ax.plot(sp, sel.train, marker="o", ms=6, mfc="none", lw=1.0, ls="--", color="C0",
                label="training accuracy, learning rate chosen per b (seed 0)")
        ax.plot(sp, sel.test, marker="s", ms=6, mfc="none", lw=1.0, ls="--", color="C1",
                label="test accuracy, learning rate chosen per b (seed 0)")
        for x0, (b, r) in zip(sp, sel.iterrows()):
            ax.annotate(f"rate {r['lr']}", (x0, r["test"]), textcoords="offset points", xytext=(5, 5) if x0 < len(bs) - 1 else (-8, 9),
                        ha="left" if x0 < len(bs) - 1 else "right", fontsize=6.5, color="C1")
        if 0.75 in where:
            ax.axvspan(where[0.75] - 0.5, len(bs) - 0.5, color="0.85", alpha=0.5, lw=0, zorder=0)
            ax.text((where[0.75] + len(bs) - 1) / 2, 0.95, "learning rate 0.5 is unstable", ha="center", fontsize=7.5, color="0.25")
    ax.axhline(MAJORITY, color="0.4", ls="--", lw=0.8, label="most frequent class")
    ax.set_xlim(-0.5, len(bs) - 0.5)
    ax.set_xticks(pos); ax.set_xticklabels([str(b) for b in bs], fontsize=8)
    ax.set_xlabel("Laplace noise scale b (pixel range is [0, 1]; equally spaced positions)"); ax.set_ylabel("accuracy after 300 rounds"); ax.set_ylim(0, 1)
    top = ax.twiny()
    top.set_xlim(ax.get_xlim()); top.set_xticks(pos[1:]); top.grid(False)
    top.set_xticklabels([f"{784 / b:.0f}" for b in bs[1:]], fontsize=7)
    top.set_xlabel("worst-case epsilon per image = 784 / b", fontsize=8)
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.33))
    fig.suptitle("Accuracy after 300 rounds against the noise scale (markers: mean of 3 seeds, bars: min-max; dashed: learning rate\n"
                 "with the highest validation accuracy among runs that did not diverge)", fontsize=9)
    fig.tight_layout()
    save(fig, "noise_accuracy.png")


def fig_gallery():
    """Non-standard plot: what a perturbed image looks like, and what the model trained at that b predicts."""
    clients, _ = data.load_raw()
    scales = [b for b in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0] if noise_paths(b)]
    sel = noise_lr_selected() if glob.glob(os.path.join(RES, "noise_lr", "*")) else None
    picks = [(0, 0), (3, 5), (7, 11), (12, 2)]
    rj = os.path.join(RES, "validation", "reident.json")
    rid = json.load(open(rj)) if os.path.exists(rj) else {}
    fig, axes = plt.subplots(len(picks) + 1, len(scales), figsize=(1.75 * len(scales), 1.75 * (len(picks) + 1) + 0.9))
    rng = np.random.default_rng(5)
    for j, b in enumerate(scales):
        chosen = sel is not None and b in sel.index       # model trained with the learning rate chosen for this b
        path, lr = (sel.loc[b, "path"], sel.loc[b, "lr"]) if chosen else (noise_paths(b)[0], 0.5)
        w = torch.load(os.path.join(path, "final_weights.pt"))
        for i, (k, idx) in enumerate(picks):
            x, y = clients[k][0][idx], clients[k][1][idx]
            xn = x + data.laplace_noise(x.shape, b, rng)
            logits = core.forward(w, torch.from_numpy(1.0 - xn)[None])
            pred = int(logits.argmax(1)); conf = float(torch.softmax(logits, 1).max())
            ax = axes[i, j]
            ax.imshow(xn.reshape(28, 28), cmap="gray", vmin=-1.0, vmax=2.0)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_title(f"{data.CLASS_NAMES[y]} read as {data.CLASS_NAMES[pred]} ({conf:.0%})", fontsize=10,
                         color="black" if pred == y else "#c44e52")
        w1 = core.unpack(w)[0]
        unit = int(w1.norm(dim=1).argmax())
        ax = axes[-1, j]
        v = float(w1[unit].abs().max())
        ax.imshow(w1[unit].reshape(28, 28), cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        extra = f"\nidentified {rid[str(b)]['ident']:.0%}" if str(b) in rid else ""
        ax.set_xlabel(f"b = {b}, rate {lr}{extra}", fontsize=10)
        ax.set_title("largest weight vector", fontsize=10)
    fig.suptitle("Rows 1-4: four images with Laplace noise of scale b (grey range -1 to 2); title: true class, the class read by the\n"
                 "model trained at that b, and its probability (red: wrong). Row 5: first-layer weight vector with the largest norm.\n"
                 "'identified': share of perturbed images an observer matches to their original among 2,000 candidates.", fontsize=10.5)
    fig.tight_layout()
    save(fig, "noise_gallery.png")

    if rid:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        bs = [b for b in NOISE_SCALES if b > 0 and str(b) in rid and noise_paths(b)]
        ident = [rid[str(b)]["ident"] for b in bs]
        test = [np.mean([summary(p)["test_acc"] for p in noise_paths(b)]) for b in bs]
        ax.plot(ident, test, "-s", ms=4, color="C1", label="learning rate 0.5 (mean of 3 seeds)")
        offsets = {0.1: (-42, -3), 0.2: (-40, -4), 0.3: (-40, -4), 0.5: (6, -8), 0.75: (2, 6), 1.5: (-2, 7)}
        for b, xi, yi in zip(bs, ident, test):
            if b in offsets:
                ax.annotate(f"b = {b}", (xi, yi), textcoords="offset points", xytext=offsets[b], fontsize=7, color="C1")
        if sel is not None:
            sb = [b for b in sel.index if str(b) in rid]
            si = [rid[str(b)]["ident"] for b in sb]
            ax.plot(si, sel.loc[sb, "test"], "--s", ms=6, mfc="none", color="C3", label="learning rate chosen per b (seed 0)")
            for b, xi in zip(sb, si):
                if b != 0.2:
                    ax.annotate(f"b = {b}", (xi, sel.loc[b, "test"]), textcoords="offset points", xytext=(8, -10), fontsize=7, color="C3")
        ax.axhline(MAJORITY, color="0.4", ls="--", lw=0.8, label="most frequent class")
        ax.set_xlabel("share of perturbed images matched to their original among 2,000 candidates")
        ax.set_ylabel("test accuracy of the model trained on perturbed images"); ax.set_ylim(0, 1); ax.set_xlim(-0.03, 1.03)
        ax.set_title("Test accuracy against re-identification, one point per noise scale b\n"
                     "(the three points at 1.0 are b = 0.02, 0.05 and 0.1)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        save(fig, "privacy_utility.png")


def diverged(h):
    """A run counts as diverged when its training loss left the finite range or exceeded 20 (ln 62 = 4.13 at the start)."""
    return bool((~np.isfinite(h.train_loss)).any() or h.train_loss.max() > 20)


def noise_lr_table():
    """One row per (b, learning rate, E) of the noise_lr sweep; lr = 0.5 with E = 5 is seed 0 of the noise sweep."""
    rows = []
    paths = [(p, re.match(r"b([\d.]+)_lr([\d.]+)(_E1)?$", os.path.basename(p))) for p in glob.glob(os.path.join(RES, "noise_lr", "*"))]
    paths = [(p, float(m.group(1)), float(m.group(2)), 1 if m.group(3) else 5) for p, m in paths if m]
    paths += [(os.path.join(RES, "noise", f"b{b}_seed0"), b, 0.5, 5) for b in sorted({q[1] for q in paths})]
    for p, b, lr, E in paths:
        if not os.path.exists(os.path.join(p, "summary.json")):
            continue
        h = hist(p)
        best = int(h.val_acc.idxmax())
        rows.append(dict(b=b, lr=lr, E=E, path=p, train=h.train_acc.iloc[-1], val=h.val_acc.iloc[-1], clean=h.clean_train_acc.iloc[-1],
                         test=summary(p)["test_acc"], diverged=diverged(h), max_train_loss=float(np.nanmax(h.train_loss)),
                         rounds_without_update=int((h.n_returned.iloc[1:] == 0).sum()), best_val=h.val_acc.max(), best_val_round=best,
                         clean_at_best_val=h.clean_train_acc.iloc[best]))
    return pd.DataFrame(rows).sort_values(["b", "E", "lr"]).reset_index(drop=True)


def noise_lr_selected():
    """For each b of the noise_lr sweep: the E = 5 run with the highest final validation accuracy among the runs that did not diverge."""
    d = noise_lr_table()
    d = d[(d.E == 5) & (~d.diverged)]
    return d.loc[d.groupby("b").val.idxmax()].set_index("b")


def fig_noise_lr():
    d = noise_lr_table()
    d.drop(columns="path").to_csv(os.path.join(FIG, "noise_lr_table.csv"), index=False)
    sel = noise_lr_selected()
    d5 = d[d.E == 5]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    div = d5.pivot(index="b", columns="lr", values="diverged")
    for ax, col, title in zip(axes, ["train", "val", "test"],
                              ["training accuracy (perturbed images)", "validation accuracy (perturbed images)", "test accuracy (clean images)"]):
        t = d5.pivot(index="b", columns="lr", values=col)
        im = ax.imshow(t.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(t.columns))); ax.set_xticklabels(t.columns)
        ax.set_yticks(range(len(t.index))); ax.set_yticklabels(t.index)
        ax.set_xlabel("client learning rate"); ax.set_ylabel("noise scale b"); ax.set_title(title, fontsize=9); ax.grid(False)
        for i, b in enumerate(t.index):
            for j, lr in enumerate(t.columns):
                if np.isnan(t.values[i, j]):
                    continue
                ax.text(j, i, f"{t.values[i, j]:.3f}" + ("\ndiverged" if div.values[i, j] else ""), ha="center", va="center",
                        fontsize=6.5, color="white", bbox=dict(facecolor="black", alpha=0.35, lw=0, pad=0.8))
                if b in sel.index and sel.loc[b, "lr"] == lr:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#d62728", lw=1.8))
        fig.colorbar(im, ax=ax)
    fig.suptitle("Learning rate re-tuned for each noise scale (C = 10%, E = 5, B = 20, 300 rounds, seed 0); "
                 "red frame = highest validation accuracy among runs that did not diverge", fontsize=9.5)
    fig.tight_layout()
    save(fig, "noise_lr_heatmap.png")


FIGS = dict(data=fig_data, main=fig_main, tune=fig_tune, ce=fig_ce, clients=fig_clients, ray=fig_ray, noise=fig_noise, noise_lr=fig_noise_lr, gallery=fig_gallery)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    for name in (a.only or FIGS):
        FIGS[name]()


if __name__ == "__main__":
    main()
