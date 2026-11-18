"""Validation experiments. Each one states a prediction and measures it.

    python -m fedavg.validate                 # all experiments
    python -m fedavg.validate --only fedsgd trace
    python -m fedavg.validate --list

Results: results/validation/<id>.md, <id>.json and figures; validation.md joins all of them.
Experiments that read sweep results need `python -m fedavg.sweep --name main|checks|noise` first.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from . import core, data
from .core import Config, aggregate, client_update, evaluate, forward, init_vector, run_fedavg, update_seed
from .sweep import BEST, NOISE_SCALES

OUT = os.path.join("results", "validation")
RES = "results"
_RAW = None


def raw():
    global _RAW
    if _RAW is None:
        _RAW = data.load_raw()
    return _RAW


def fed(**kw):
    return data.build_federation(raw=raw(), **kw)


def best_cfg(**kw):
    base = dict(rounds=BEST["rounds"], clients_per_round=10, epochs=BEST["E"], batch_size=BEST["B"], lr=BEST["lr"], seed=0)
    base.update(kw)
    return Config(**base)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def save(name, title, md, numbers):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write(f"## {title}\n\n{md}\n")
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(f"[{name}] done", flush=True)


def load_run(path):
    h = pd.read_csv(os.path.join(path, "history.csv"))
    s = json.load(open(os.path.join(path, "summary.json")))
    a = np.load(os.path.join(path, "arrays.npz"))
    return h, s, a


# ------------------------------------------------------------------ Part 1.1
def fedsgd():
    """F1. All 100 clients, E = 1, full batch: FedAvg must equal gradient descent on the pooled data."""
    num = {}
    rows = []
    for name, dtype in [("float64", torch.float64), ("float32", torch.float32)]:
        f = fed(dtype=dtype)
        cfg = Config(rounds=20, clients_per_round=100, epochs=1, batch_size=0, lr=0.5, seed=0)
        w, h = run_fedavg(f, cfg)
        ref = init_vector(0, dtype).requires_grad_(True)
        for _ in range(20):
            (g,) = torch.autograd.grad(F.cross_entropy(forward(ref, f.x_tr), f.y_tr), ref)
            with torch.no_grad():
                ref -= 0.5 * g
        ref = ref.detach()
        wu, _ = run_fedavg(f, Config(**{**cfg.__dict__, "weighting": "uniform"}))
        moved = float((ref - init_vector(0, dtype)).abs().max())
        num[name] = dict(max_abs_diff=float((w - ref).abs().max()), uniform_max_abs_diff=float((wu - ref).abs().max()),
                         largest_weight_change=moved, loss_fedavg=float(h["train_loss"][-1]),
                         loss_centralized=float(F.cross_entropy(forward(ref, f.x_tr), f.y_tr)))
        n = num[name]
        rows.append([name, f"{n['max_abs_diff']:.2e}", f"{n['uniform_max_abs_diff']:.2e}", f"{moved:.3f}",
                     f"{n['loss_fedavg']:.6f}", f"{n['loss_centralized']:.6f}"])
    md = ("Prediction: with all 100 clients, one local epoch and full-batch steps, the sample-weighted average of the "
          "client models equals one gradient-descent step on the pooled training loss. 20 rounds, learning rate 0.5.\n\n"
          + table(["precision", "max abs weight difference, FedAvg vs centralized", "same with 1/m weights",
                   "largest weight change from the start", "FedAvg training loss", "centralized training loss"], rows))
    save("fedsgd", "F1. FedSGD limit equals centralized gradient descent", md, num)


def trace():
    """F2. First 10 rounds, with the server step recomputed independently in numpy."""
    f = fed()
    cfg = best_cfg(rounds=10)
    rows, detail, num = [], [], dict(recompute_err=[], sum_p=[])

    def hook(rnd, w_old, results, w_new):
        ns = np.array([r["stats"]["n"] for r in results], dtype=np.float64)
        p = ns / ns.sum()
        manual = np.zeros(core.N_PARAMS)
        for pk, r in zip(p, results):
            manual += pk * r["vec"].double().numpy()
        err = float(np.abs(manual - w_new.double().numpy()).max())
        before = evaluate(w_old, f.x_tr, f.y_tr)
        after = evaluate(w_new, f.x_tr, f.y_tr)
        va = evaluate(w_new, f.x_va, f.y_va)
        lf = float(np.dot(p, [r["stats"]["loss_first"] for r in results]))
        ll = float(np.dot(p, [r["stats"]["loss_last"] for r in results]))
        local = float(np.dot(p, [float((r["vec"] - w_old).norm()) for r in results]))
        step = float((w_new - w_old).norm())
        num["recompute_err"].append(err); num["sum_p"].append(float(p.sum()))
        rows.append([rnd, " ".join(str(r["cid"]) for r in results), int(ns.sum()), f"{lf:.3f} to {ll:.3f}",
                     f"{before['loss']:.3f} to {after['loss']:.3f}", f"{after['acc']:.3f}", f"{va['acc']:.3f}",
                     f"{local:.2f}", f"{step:.2f}", f"{step / local:.2f}", f"{err:.1e}"])
        if rnd == 1:
            for pk, r in zip(p, results):
                x, y = f.client_train(r["cid"])
                own = evaluate(r["vec"], x, y)
                glob_ = evaluate(w_new, x, y)["acc"]
                detail.append([r["cid"], r["stats"]["n"], f"{pk:.4f}", r["stats"]["steps"], f"{r['stats']['loss_first']:.3f}",
                               f"{r['stats']['loss_last']:.3f}", f"{own['loss']:.3f}", f"{own['acc']:.3f}", f"{glob_:.3f}"])
            num["round1_loss_first"] = [r["stats"]["loss_first"] for r in results]
            num["round1_steps_ok"] = all(r["stats"]["steps"] == cfg.epochs * int(np.ceil(r["stats"]["n"] / cfg.batch_size)) for r in results)

    run_fedavg(f, cfg, on_round=hook)
    num["ln62"] = float(np.log(62))
    md = (f"Settings: 10 clients per round, E = {cfg.epochs}, B = {cfg.batch_size}, learning rate {cfg.lr}, seed 0. "
          "The last column is the largest absolute difference between the server's new weights and an independent "
          "numpy recomputation of sum_k (n_k / n) w_k.\n\n"
          + table(["round", "selected clients", "their samples", "local loss, first step to last epoch",
                   "global training loss, before to after", "training accuracy", "validation accuracy",
                   "mean client move", "server move", "ratio", "recomputation error"], rows)
          + f"\nRound 1 in detail. Every client starts from the same initial weights, whose predictions are close to "
            f"uniform, so the first minibatch loss must be close to ln 62 = {np.log(62):.3f}.\n\n"
          + table(["client", "n_k", "weight n_k / n", "SGD steps", "loss at first step", "mean loss in last epoch",
                   "loss of its final model on its data", "accuracy of its final model on its data", "accuracy of the new global model on its data"], detail))
    save("trace", "F2. The first 10 rounds step by step", md, num)


def references():
    """F5. Majority class, local-only training and centralized training bracket FedAvg."""
    f = fed()
    K = f.n_clients
    num = {}
    # majority-class predictors
    counts = torch.bincount(f.y_tr, minlength=62)
    num["majority_val_acc"] = float((f.y_va == counts.argmax()).double().mean())
    num["majority_test_acc"] = float((f.y_te == counts.argmax()).double().mean())
    hits = 0
    for k in range(K):
        yk = f.client_train(k)[1]
        hits += int((f.client_val(k)[1] == torch.bincount(yk, minlength=62).argmax()).sum())
    num["own_majority_val_acc"] = hits / len(f.y_va)
    # centralized minibatch SGD on the pooled training data
    cen = {}
    for lr in [0.05, 0.1, 0.2, 0.5]:
        w = init_vector(0)
        acc = []
        for ep in range(1, 61):
            w, _ = client_update(w, f.x_tr, f.y_tr, 1, BEST["B"], lr, update_seed(0, ep, 10 ** 6))
            acc.append(evaluate(w, f.x_va, f.y_va)["acc"])
        cen[lr] = dict(val_acc=acc, test_acc=evaluate(w, f.x_te, f.y_te)["acc"], train_acc=evaluate(w, f.x_tr, f.y_tr)["acc"])
    best_lr = max(cen, key=lambda lr: max(cen[lr]["val_acc"]))
    num["centralized"] = {str(k): dict(best_val=max(v["val_acc"]), best_epoch=int(np.argmax(v["val_acc"])) + 1,
                                       final_val=v["val_acc"][-1], final_test=v["test_acc"], final_train=v["train_acc"])
                          for k, v in cen.items()}
    num["centralized_best_lr"] = best_lr
    # local-only: every client trains alone from the same initial weights
    loc, loc_unseen = {}, {}
    checkpoints = [1, 2, 5, 10, 20, 40, 70, 100]
    for lr in [0.1, 0.5]:
        correct = np.zeros((len(checkpoints), K))
        unseen = np.zeros(len(checkpoints))
        for k in range(K):
            x, y = f.client_train(k)
            xv, yv = f.client_val(k)
            absent = ~torch.isin(yv, y)      # validation images of a class the client has no training image of
            w, done = init_vector(0), 0
            for i, ep in enumerate(checkpoints):
                w, _ = client_update(w, x, y, ep - done, BEST["B"], lr, update_seed(0, ep, k))
                done = ep
                hit = forward(w, xv).argmax(1) == yv
                correct[i, k] = int(hit.sum())
                unseen[i] += int(hit[absent].sum())
        loc[lr] = correct
        loc_unseen[lr] = unseen
    pooled = {lr: c.sum(1) / len(f.y_va) for lr, c in loc.items()}
    lr_loc = max(pooled, key=lambda lr: pooled[lr].max())
    i_best = int(pooled[lr_loc].argmax())
    num["local_only"] = dict(lr=lr_loc, epochs=checkpoints[i_best], val_acc=float(pooled[lr_loc][i_best]),
                             by_epochs={str(e): float(a) for e, a in zip(checkpoints, pooled[lr_loc])})
    md_fed = ""
    main = os.path.join(RES, "main", "seed0")
    if os.path.exists(os.path.join(main, "arrays.npz")):
        h, s, a = load_run(main)
        fed_client = a["client_val_acc"][-1]
        loc_client = loc[lr_loc][i_best] / f.n_va
        num["fedavg_val_acc"] = float(h.val_acc.iloc[-1]); num["fedavg_test_acc"] = s["test_acc"]
        w_fed = torch.load(os.path.join(main, "final_weights.pt"))
        absent = torch.cat([~torch.isin(f.client_val(k)[1], f.client_train(k)[1]) for k in range(K)])
        hit = forward(w_fed, f.x_va).argmax(1) == f.y_va
        num["absent_class_images"] = int(absent.sum())
        num["absent_class_acc_fedavg"] = float(hit[absent].double().mean())
        num["present_class_acc_fedavg"] = float(hit[~absent].double().mean())
        num["absent_class_acc_local"] = float(loc_unseen[lr_loc][i_best] / int(absent.sum()))
        num["clients_better_with_fedavg"] = int((fed_client > loc_client).sum())
        num["clients_equal"] = int((fed_client == loc_client).sum())
        num["min_gain"] = float((fed_client - loc_client).min()); num["median_gain"] = float(np.median(fed_client - loc_client))
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        sc = ax.scatter(loc_client, fed_client, c=f.n_tr, cmap="viridis", s=18)
        ax.plot([0, 1], [0, 1], color="0.5", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("validation accuracy, client trained alone"); ax.set_ylabel("validation accuracy, FedAvg global model")
        ax.set_title("One point per client (100 clients)")
        fig.colorbar(sc, label="training samples of the client")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "local_vs_fedavg.png"), dpi=150); plt.close(fig)
        md_fed = (f"FedAvg (seed 0, round {len(h) - 1}) reaches {num['fedavg_val_acc']:.4f}. Its global model beats the "
                  f"client's own model on {num['clients_better_with_fedavg']} of 100 clients "
                  f"(median gain {num['median_gain']:.3f}, smallest gain {num['min_gain']:.3f}).\n\n"
                  f"{num['absent_class_images']} of the {len(f.y_va)} validation images belong to a class of which their "
                  f"client has no training image. A client trained alone classifies {num['absent_class_acc_local']:.3f} of "
                  f"them correctly; the FedAvg model classifies {num['absent_class_acc_fedavg']:.3f} "
                  f"({num['present_class_acc_fedavg']:.3f} on the other validation images).\n")
    c = num["centralized"][str(best_lr)]
    md = ("Prediction: a correct FedAvg model must score above both majority-class predictors and above clients "
          "trained alone, and at most slightly above centralized training of the same network on the pooled data.\n\n"
          + table(["reference", "pooled validation accuracy"], [
              ["most frequent class of the pooled training data", f"{num['majority_val_acc']:.4f}"],
              ["each client's own most frequent class", f"{num['own_majority_val_acc']:.4f}"],
              [f"each client trained alone (best of 2 learning rates and 8 epoch counts: lr {lr_loc}, {checkpoints[i_best]} epochs)",
               f"{num['local_only']['val_acc']:.4f}"],
              [f"centralized SGD on pooled data (best of 4 learning rates: {best_lr}, best epoch {c['best_epoch']} of 60)", f"{c['best_val']:.4f}"],
              [f"centralized SGD, same run after 60 epochs", f"{c['final_val']:.4f}"]])
          + "\n" + md_fed)
    save("references", "F5. Reference models that bracket FedAvg", md, num)


def sampling():
    """F6. Client selection is uniform without replacement."""
    rows, num = [], {}
    for path in sorted(glob.glob(os.path.join(RES, "main", "seed*"))):
        sel = np.load(os.path.join(path, "arrays.npz"))["selected"][1:]
        R, K = sel.shape
        m = int(sel[0].sum())
        c = sel.sum(0)
        exp = R * m / K
        chi = float(((c - exp) ** 2 / (exp * (1 - m / K))).sum())
        pval = float(stats.chi2.sf(chi, K - 1))
        seen = np.cumsum(sel, axis=0) > 0
        full = int(np.argmax(seen.all(1))) + 1 if seen.all(1).any() else -1
        unseen10 = int(K - seen[9].sum())
        num[os.path.basename(path)] = dict(min=int(c.min()), max=int(c.max()), chi2=chi, p=pval, covered_by=full,
                                           unseen_after_10=unseen10, per_round_ok=bool((sel.sum(1) == m).all()))
        rows.append([os.path.basename(path), bool((sel.sum(1) == m).all()), f"{exp:.0f}", int(c.min()), int(c.max()),
                     f"{chi:.1f}", f"{pval:.2f}", unseen10, full])
    md = ("Prediction: every round has exactly 10 distinct clients; a client's number of selections in 300 rounds is "
          "Binomial(300, 0.1) with mean 30 and standard deviation 5.2; the expected number of clients not yet selected "
          f"after 10 rounds is 100 x 0.9^10 = {100 * 0.9 ** 10:.1f}. The chi-square statistic has 99 degrees of freedom.\n\n"
          + table(["run", "10 distinct clients in every round", "expected selections", "fewest", "most",
                   "chi-square", "p-value", "clients unseen after 10 rounds", "round by which all were selected"], rows))
    save("sampling", "F6. Client sampling statistics", md, num)


def structure():
    """F3, F4, F7, F8. Runs that change the structure of the problem (sweep 'checks')."""
    base = os.path.join(RES, "checks")
    num = {}
    rows = []
    for E in [1, 5, 20]:
        for part in ["writer", "iid"]:
            vals, tests, align, spread, r70 = [], [], [], [], []
            for path in sorted(glob.glob(os.path.join(base, f"{part}_E{E}_seed*"))):
                h, s, a = load_run(path)
                vals.append(h.val_acc.iloc[-1]); tests.append(s["test_acc"])
                align.append(float((h.update_norm / h.local_norm)[1:51].mean()))
                spread.append(float(a["client_val_acc"][-1].std()))
                hit = np.flatnonzero(h.val_acc.values >= 0.70)
                r70.append(int(hit[0]) if len(hit) else -1)
            if not vals:
                continue
            num[f"{part}_E{E}"] = dict(val=vals, test=tests, align=align, spread=spread, rounds_to_70=r70)
            rows.append([E, "real writers" if part == "writer" else "random (IID) deal", f"{np.mean(vals):.4f}",
                         f"{np.mean(tests):.4f}", f"{np.mean(align):.3f}", f"{np.mean(spread):.3f}", " and ".join(map(str, r70))])
    md = ("**F4. Writers against a random deal of the same images.** The 'IID' federation deals the same 31,825 images "
          "to 100 clients at random with the original client sizes. Prediction: with identically distributed clients "
          "the local updates point in more similar directions, so the server step is a larger fraction of the mean "
          "client step, and the clients' validation accuracies differ less. If 10 equally weighted client steps were "
          "orthogonal the ratio would be 1/sqrt(10) = 0.32; if they were identical it would be 1.\n\n"
          + table(["E", "partition", "validation accuracy, round 300", "test accuracy", "server step / mean client step, rounds 1-50",
                   "std of per-client validation accuracy", "rounds to 70% validation accuracy (2 seeds)"], rows))
    p = os.path.join(base, "shuffled_labels")
    if os.path.exists(p):
        h, s, a = load_run(p)
        num["shuffled"] = dict(val_max=float(h.val_acc.max()), val_final=float(h.val_acc.iloc[-1]),
                               train_final=float(h.train_acc.iloc[-1]), test=s["test_acc"], train_loss=float(h.train_loss.iloc[-1]))
        md += ("\n**F3. Shuffled labels.** The training labels are permuted inside each client; validation labels are "
               "untouched. No image carries information about its label, so validation accuracy must stay near the "
               f"majority-class rate (5.8%). Measured: highest validation accuracy over 300 rounds {h.val_acc.max():.4f}, "
               f"final {h.val_acc.iloc[-1]:.4f}, test {s['test_acc']:.4f}; training accuracy on the shuffled labels "
               f"{h.train_acc.iloc[-1]:.4f}; training loss {h.train_loss.iloc[-1]:.3f} (ln 62 = 4.127).\n")
    rows = []
    for prep in ["invert", "raw", "center"]:
        for lr in [0.05, 0.5]:
            p = os.path.join(base, f"prep_{prep}_lr{lr}")
            if os.path.exists(p):
                h, s, _ = load_run(p)
                num[f"prep_{prep}_lr{lr}"] = dict(val=float(h.val_acc.iloc[-1]), val50=float(h.val_acc[50]), test=s["test_acc"])
                rows.append([prep, lr, f"{h.val_acc[50]:.4f}", f"{h.val_acc.iloc[-1]:.4f}", f"{h.train_acc.iloc[-1]:.4f}", f"{s['test_acc']:.4f}"])
    if rows:
        md += ("\n**F7. Input scaling.** 'invert' feeds 1 - x (ink 1, page 0), 'raw' feeds x (page 1), 'center' feeds "
               "1 - 2x (page -1, ink 1). The three inputs are affine maps of each other, so the network can represent "
               "the same functions; any difference comes from the optimisation.\n\n"
               + table(["input", "learning rate", "validation accuracy, round 50", "round 300", "training accuracy", "test accuracy"], rows))
    uni = [load_run(p) for p in sorted(glob.glob(os.path.join(base, "uniform_seed*")))]
    main = [load_run(p) for p in sorted(glob.glob(os.path.join(RES, "main", "seed[0-2]")))]
    if uni and main:
        u = [h.val_acc.iloc[-1] for h, _, _ in uni]; w_ = [h.val_acc.iloc[-1] for h, _, _ in main]
        num["uniform"] = dict(val=u, weighted_val=w_)
        md += ("\n**F8. Weights n_k / n against 1 / m on the real data.** Client sizes range from 94 to 314 training "
               "images, so the two rules differ little here (test F1 shows a case where only n_k / n is correct). "
               f"Validation accuracy at round 300 over seeds 0-2: n_k / n {np.mean(w_):.4f} "
               f"({', '.join(f'{v:.4f}' for v in w_)}); 1 / m {np.mean(u):.4f} ({', '.join(f'{v:.4f}' for v in u)}).\n")
    rows = []
    for d in ["0.995", "0.99"]:
        p = os.path.join(base, f"lrdecay{d}")
        if os.path.exists(p):
            h, s, _ = load_run(p)
            num[f"lrdecay{d}"] = dict(val=float(h.val_acc.iloc[-1]), val_loss=float(h.val_loss.iloc[-1]), test=s["test_acc"])
            rows.append([d, f"{h.val_acc.iloc[-1]:.4f}", f"{h.val_loss.iloc[-1]:.4f}", f"{h.train_acc.iloc[-1]:.4f}", f"{s['test_acc']:.4f}"])
    if rows and main:
        h, s, _ = main[0]
        rows.insert(0, ["1 (constant)", f"{h.val_acc.iloc[-1]:.4f}", f"{h.val_loss.iloc[-1]:.4f}", f"{h.train_acc.iloc[-1]:.4f}", f"{s['test_acc']:.4f}"])
        md += ("\n**F9. Learning-rate decay per round (seed 0).**\n\n"
               + table(["decay factor per round", "validation accuracy", "validation loss", "training accuracy", "test accuracy"], rows))
    save("structure", "F3, F4, F7, F8, F9. Runs that change the structure of the problem", md, num)


def errors():
    """F10. Which test errors remain, and the 18 test images that also occur in the training data."""
    f = fed()
    clients, (xt_raw, yt) = raw()
    num, md = {}, ""
    fold = lambda c: np.where(c < 10, c, 10 + (c - 10) % 26)
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, "main", "seed*"))):
        w = torch.load(os.path.join(path, "final_weights.pt"))
        pred = forward(w, f.x_te).argmax(1).numpy()
        y = f.y_te.numpy()
        wrong = pred != y
        case = wrong & (fold(pred) == fold(y))
        g0, g1 = {0, 24, 50}, {1, 18, 47}   # {0, O, o} and {1, I, l}
        dig = wrong & ~case & ((np.isin(pred, list(g0)) & np.isin(y, list(g0))) | (np.isin(pred, list(g1)) & np.isin(y, list(g1))))
        tr_hash = {hashlib.md5(a.tobytes()).hexdigest() for x, _ in clients for a in x}
        dup = np.array([hashlib.md5(a.tobytes()).hexdigest() in tr_hash for a in xt_raw])
        d = dict(acc=float((~wrong).mean()), case_insensitive_acc=float((fold(pred) == fold(y)).mean()),
                 errors=int(wrong.sum()), case_errors=int(case.sum()), digit_letter_errors=int(dig.sum()),
                 n_dup=int(dup.sum()), acc_without_dup=float((~wrong)[~dup].mean()), acc_on_dup=float((~wrong)[dup].mean()))
        num[os.path.basename(path)] = d
        rows.append([os.path.basename(path), f"{d['acc']:.4f}", d["errors"], f"{d['case_errors']} ({d['case_errors'] / d['errors']:.0%})",
                     f"{d['digit_letter_errors']} ({d['digit_letter_errors'] / d['errors']:.0%})", f"{d['case_insensitive_acc']:.4f}",
                     f"{d['acc_without_dup']:.4f}", f"{d['acc_on_dup']:.3f}"])
        if path.endswith("seed0"):
            conf = np.zeros((62, 62), dtype=int)
            np.add.at(conf, (y, pred), 1)
            off = conf.copy(); np.fill_diagonal(off, 0)
            sym = off + off.T
            pairs = sorted(((sym[i, j], i, j) for i in range(62) for j in range(i + 1, 62)), reverse=True)[:12]
            num["top_pairs"] = [(data.CLASS_NAMES[i], data.CLASS_NAMES[j], int(c)) for c, i, j in pairs]
            per_class = conf.diagonal() / np.maximum(conf.sum(1), 1)
            train_counts = np.bincount(np.concatenate([yy for _, yy in clients]), minlength=62)
            rho = stats.spearmanr(train_counts, per_class)
            num["spearman_class_count_vs_acc"] = [float(rho.statistic), float(rho.pvalue)]
            np.save(os.path.join(OUT, "confusion_seed0.npy"), conf)
    md = ("Prediction: handwritten letters such as c/C, o/O, s/S differ between cases mainly in size, which a 28 x 28 "
          "crop removes, and 0/O/o and 1/I/l share one shape. If the model is otherwise correct, a large share of its "
          "test errors must be such pairs, and accuracy with upper and lower case merged must be much higher.\n\n"
          + table(["model", "test accuracy", "errors of 3,621", "same letter, other case", "within {0, O, o} or {1, I, l}",
                   "accuracy with cases merged", "accuracy without the 18 duplicated images", "accuracy on the 18"], rows))
    # validation images that also occur, with the same label, in a training split
    key = lambda x, y: [hashlib.md5(a.tobytes()).hexdigest() + f"_{int(c)}" for a, c in zip(x.numpy(), y.numpy())]
    tr_k, va_k = key(f.x_tr, f.y_tr), key(f.x_va, f.y_va)
    all_tr = set(tr_k)
    own, other = np.zeros(len(va_k), dtype=bool), np.zeros(len(va_k), dtype=bool)
    for k in range(f.n_clients):
        mine = set(tr_k[f.off_tr[k]:f.off_tr[k + 1]])
        for i in range(f.off_va[k], f.off_va[k + 1]):
            own[i] = va_k[i] in mine
            other[i] = (va_k[i] in all_tr) and not own[i]
    dup_va = own | other
    vrows = []
    for path in sorted(glob.glob(os.path.join(RES, "main", "seed*"))):
        w = torch.load(os.path.join(path, "final_weights.pt"))
        ok = (forward(w, f.x_va).argmax(1) == f.y_va).numpy()
        d = dict(val_acc=float(ok.mean()), val_acc_without_dup=float(ok[~dup_va].mean()), val_acc_on_dup=float(ok[dup_va].mean()))
        num[os.path.basename(path)].update(d)
        vrows.append([os.path.basename(path), f"{d['val_acc']:.4f}", f"{d['val_acc_without_dup']:.4f}", f"{d['val_acc_on_dup']:.3f}"])
    num["val_dup_same_client"], num["val_dup_other_client"], num["val_n"] = int(own.sum()), int(other.sum()), int(len(va_k))
    md += (f"\nThe random 80/20 split does not remove duplicates. {int(own.sum())} of the {len(va_k)} validation images have a "
           f"pixel-identical copy with the same label in the training split of the same client, and {int(other.sum())} more "
           "have one at another client only. A model that had memorised its training images would classify these "
           "images almost perfectly. Removing them can change validation accuracy by at most "
           f"{dup_va.sum() / len(va_k):.4f}.\n\n"
           + table(["model", "validation accuracy", "without the duplicated images", "on the duplicated images"], vrows))
    if "top_pairs" in num:
        md += ("\nMost confused pairs for seed 0 (both directions added): "
               + ", ".join(f"{a}/{b} {c}" for a, b, c in num["top_pairs"]) + ". "
               f"Spearman correlation between a class's number of training images and its test accuracy: "
               f"{num['spearman_class_count_vs_acc'][0]:.2f} (p = {num['spearman_class_count_vs_acc'][1]:.1e}).\n")
    save("errors", "F10. Structure of the remaining test errors and duplicated images", md, num)


def drift():
    """F11. Client models and the server average in the plane of the two largest principal components."""
    f = fed()
    num = {}
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), gridspec_kw=dict(wspace=0.28))
    for ax, E in zip(axes, [1, 5, 20]):
        cfg = best_cfg(rounds=8, epochs=E)
        segs = []
        run_fedavg(f, cfg, on_round=lambda rnd, w_old, results, w_new: segs.append(
            (w_old.double().numpy(), [r["vec"].double().numpy() for r in results], w_new.double().numpy())))
        pts = np.array([s[0] for s in segs] + [segs[-1][2]] + [c for s in segs for c in s[1]])
        mean = pts.mean(0)
        _, sv, vt = np.linalg.svd(pts - mean, full_matrices=False)
        proj = lambda v: (np.asarray(v) - mean) @ vt[:2].T
        explained = float((sv[:2] ** 2).sum() / (sv ** 2).sum())
        ratios = []
        cmap = plt.get_cmap("viridis")
        for i, (w_old, cl, w_new) in enumerate(segs):
            a, b = proj(w_old), proj(w_new)
            col = cmap(i / (len(segs) - 1))
            for c in cl:
                pc = proj(c)
                ax.plot([a[0], pc[0]], [a[1], pc[1]], color=col, lw=0.8, alpha=0.8)
                ax.plot(pc[0], pc[1], "o", color=col, ms=3)
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
            ratios.append(np.linalg.norm(w_new - w_old) / np.mean([np.linalg.norm(c - w_old) for c in cl]))
        g = proj(pts[:len(segs) + 1])
        ax.plot(g[:, 0], g[:, 1], "s", color="black", ms=4)
        ax.set_title(f"E = {E}\nserver step / mean client step = {np.mean(ratios):.2f}\n2 components hold {explained:.0%} of the variance", fontsize=9)
        ax.set_xlabel("principal component 1"); ax.set_ylabel("principal component 2")
        ax.set_aspect("equal", adjustable="datalim")
        num[f"E{E}"] = dict(ratio=float(np.mean(ratios)), explained=explained)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, 8))
    fig.colorbar(sm, ax=axes, label="round", fraction=0.02)
    fig.suptitle("First 8 rounds in weight space: black squares = global model, coloured lines = the 10 client updates of a round", fontsize=11, y=1.08)
    fig.savefig(os.path.join(OUT, "drift_pca.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    md = ("Non-standard plot `drift_pca.png`. Each panel projects the 108,478-dimensional weights of the first 8 rounds "
          "onto their two largest principal components. Prediction: more local epochs move each client further "
          "toward its own optimum, so the client end points spread more and the server step becomes a smaller "
          "fraction of the mean client step.\n\n"
          + table(["E", "server step / mean client step, rounds 1-8", "variance in the plotted plane"],
                  [[E, f"{num[f'E{E}']['ratio']:.3f}", f"{num[f'E{E}']['explained']:.0%}"] for E in [1, 5, 20]]))
    save("drift", "F11. Client drift in weight space", md, num)


# ------------------------------------------------------------------ Part 1.2 (Ray)
def _max_overlap(starts, ends):
    events = sorted([(s, 1) for s in starts] + [(e, -1) for e in ends], key=lambda t: (t[0], t[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def ray_equal():
    """R1, R2. The Ray run equals the sequential run bit for bit, and its actors ran concurrently."""
    path = os.path.join(RES, "ray", "seed0")
    h, s, a = load_run(path)
    f = fed()
    c = s["config"]
    cfg = Config(**c)
    w_seq, h_seq = run_fedavg(f, cfg)
    w_ray = torch.load(os.path.join(path, "final_weights.pt"))
    t = a["timings"]
    num = dict(equal_weights=bool(torch.equal(w_seq, w_ray)), max_abs_diff=float((w_seq - w_ray).abs().max()),
               equal_history=bool(np.array_equal(h_seq["val_loss"], h.val_loss.values) and np.array_equal(h_seq["train_acc"], h.train_acc.values)),
               updates=int(len(t)), distinct_pids=int(len(set(t[:, 2]))), ray_seconds=s["wall_seconds"])
    conc_train, conc_alive = [], []
    for rnd in range(1, cfg.rounds + 1):
        r = t[t[:, 0] == rnd]
        conc_train.append(_max_overlap(r[:, 3], r[:, 4])); conc_alive.append(_max_overlap(r[:, 5], r[:, 4]))
    num["rounds_with_4_training_at_once"] = int(np.sum(np.array(conc_train) == 4))
    num["max_training_at_once"] = int(max(conc_train))
    num["rounds_with_4_alive_at_once"] = int(np.sum(np.array(conc_alive) == 4))
    num["mean_actor_lifetime"] = float((t[:, 4] - t[:, 5]).mean())
    num["max_actors_alive_at_once"] = int(_max_overlap(t[:, 5], t[:, 4]))
    num["mean_train_seconds_per_update"] = float((t[:, 4] - t[:, 3]).mean())
    num["mean_round_seconds"] = float(h.seconds[1:].mean())
    t0 = time.time(); run_fedavg(f, Config(**{**c, "rounds": 30})); num["sequential_seconds_per_round"] = (time.time() - t0) / 30
    md = (f"Run `results/ray/seed0`: 4 clients per round, {cfg.rounds} rounds, E = {cfg.epochs}, B = {cfg.batch_size}, "
          f"learning rate {cfg.lr}. Prediction: a client update depends only on the global weights, the client's data "
          "and a seed derived from (run seed, round, client), so the actors must return exactly what the sequential loop returns.\n\n"
          + table(["quantity", "value"], [
              ["final weights identical to the sequential run (torch.equal)", num["equal_weights"]],
              ["largest absolute weight difference", f"{num['max_abs_diff']:.1e}"],
              ["per-round validation loss and training accuracy identical", num["equal_history"]],
              ["client updates / distinct worker process ids", f"{num['updates']} / {num['distinct_pids']}"],
              ["rounds in which all 4 actors were alive at the same time", f"{num['rounds_with_4_alive_at_once']} of {cfg.rounds}"],
              ["rounds in which all 4 training calls overlapped in time", f"{num['rounds_with_4_training_at_once']} of {cfg.rounds}"],
              ["mean seconds from actor construction to its result", f"{num['mean_actor_lifetime']:.2f}"],
              ["largest number of actors alive at one time", num["max_actors_alive_at_once"]],
              ["mean seconds per round with Ray", f"{num['mean_round_seconds']:.2f}"],
              ["mean seconds of one client's training inside its actor", f"{num['mean_train_seconds_per_update']:.3f}"],
              ["seconds per round, sequential loop", f"{num['sequential_seconds_per_round']:.3f}"]]))
    save("ray_equal", "R1, R2. Ray actors reproduce the sequential run and run concurrently", md, num)


def ray_faults():
    """R3. Crashed, slow and diverged clients, and a round in which every client fails."""
    from .ray_clients import RayRunner
    from tests.test_fedavg import DroppingRunner
    f = fed()
    cfg = Config(rounds=6, clients_per_round=4, epochs=BEST["E"], batch_size=BEST["B"], lr=BEST["lr"], seed=0)
    _, h0 = run_fedavg(f, cfg)
    sel = {r: [int(k) for k in np.flatnonzero(h0["selected"][r])] for r in range(1, 7)}
    faults = {(2, sel[2][0]): "crash", (3, sel[3][1]): "sleep:60", (4, sel[4][2]): "nan"}
    faults.update({(5, k): "crash" for k in sel[5]})
    runner = RayRunner(f, cfg, timeout=15.0, faults=faults)
    try:
        t0 = time.time()
        w_ray, h = run_fedavg(f, cfg, runner=runner)
        secs = time.time() - t0
    finally:
        runner.close()
    w_ref, h_ref = run_fedavg(f, cfg, runner=DroppingRunner(f, cfg, set(faults)))
    num = dict(returned=[int(v) for v in h["n_returned"]], equal_to_reference=bool(torch.equal(w_ray, w_ref)),
               finite=bool(torch.isfinite(w_ray).all()), round_seconds=[float(v) for v in h["seconds"]],
               val_acc=[float(v) for v in h["val_acc"]], total_seconds=secs)
    what = {1: "none", 2: "one actor process exits during training", 3: "one actor sleeps 60 s; the round waits 15 s",
            4: "one actor returns non-finite weights", 5: "all four actor processes exit", 6: "none"}
    rows = [[r, what[r], f"{int(h['n_returned'][r])} of 4", f"{h['seconds'][r]:.1f}", f"{h['val_acc'][r]:.4f}",
             "unchanged" if np.isnan(h["update_norm"][r]) else f"{h['update_norm'][r]:.2f}"] for r in range(1, 7)]
    md = ("Prediction: the server averages the clients that returned finite weights, with weights renormalised over "
          "them, and keeps the old model when nobody returned. The reference is a sequential run that omits the same "
          "(round, client) pairs.\n\n"
          + table(["round", "injected fault", "updates used", "seconds", "validation accuracy", "server step"], rows)
          + f"\nFinal weights identical to the reference run: {num['equal_to_reference']}. All weights finite: {num['finite']}.\n")
    save("ray_faults", "R3. Client failures during a round", md, num)


def ray_scaling():
    """R4. Resource allocation: with c CPUs at most c actors train at once; the result never changes."""
    from .ray_clients import RayRunner
    f = fed()
    num, rows = {}, []
    for E in [5, 100]:
        cfg = Config(rounds=4, clients_per_round=4, epochs=E, batch_size=BEST["B"], lr=0.05, seed=0)
        t0 = time.time(); w_seq, _ = run_fedavg(f, cfg); seq = (time.time() - t0) / cfg.rounds
        for cpus in [1, 2, 4]:
            runner = RayRunner(f, cfg, num_cpus=cpus)
            try:
                run_fedavg(f, Config(**{**cfg.__dict__, "rounds": 1}), runner=runner)   # warm-up round
                w, h = run_fedavg(f, cfg, runner=runner)
            finally:
                runner.close()
            t = h["timings"]
            conc = max(_max_overlap(t[t[:, 0] == r][:, 3], t[t[:, 0] == r][:, 4]) for r in range(1, cfg.rounds + 1))
            alive = _max_overlap(t[:, 5], t[:, 4])
            d = dict(seq=seq, ray=float(h["seconds"][1:].mean()), train=float((t[:, 4] - t[:, 3]).mean()),
                     max_training=int(conc), max_alive=int(alive), equal=bool(torch.equal(w, w_seq)))
            num[f"E{E}_cpus{cpus}"] = d
            rows.append([E, cpus, f"{seq:.2f}", f"{d['ray']:.2f}", f"{d['train']:.2f}", conc, alive, d["equal"]])
    md = ("Each actor reserves 1 CPU, so Ray may run at most as many actors at once as it has CPUs. Prediction: the "
          "largest number of simultaneously training actors equals min(4, CPUs), the weights are identical in every "
          "setting, and parallel execution pays off only when one client's training time exceeds the cost of creating its actor.\n\n"
          + table(["E", "CPUs given to Ray", "sequential s/round", "Ray s/round", "training s per client",
                   "most actors training at once", "most actors alive at once", "weights equal to sequential"], rows))
    save("ray_scaling", "R4. CPU allocation and wall-clock time", md, num)


def _pid_exists(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ray_memory():
    """R11. Memory of one actor process, and whether the actor processes are gone after their round."""
    import subprocess
    import ray
    from .ray_clients import ClientActor, RayRunner
    f = fed()
    cfg = Config(rounds=20, clients_per_round=4, epochs=BEST["E"], batch_size=BEST["B"], lr=BEST["lr"], seed=0)
    runner = RayRunner(f, cfg, use_gpu=False)
    now, prev, gone = [], [], []

    def hook(rnd, w_old, results, w_new):  # called after the round's actors were killed (ray.kill does not wait)
        mine = [u["pid"] for u in runner.log if u["round"] == rnd]
        now.append(sum(_pid_exists(p) for p in mine))
        prev.append(sum(_pid_exists(u["pid"]) for u in runner.log if u["round"] < rnd))
        t0 = time.time()
        while any(_pid_exists(p) for p in mine) and time.time() - t0 < 5.0:
            time.sleep(0.002)
        gone.append(time.time() - t0)

    try:
        run_fedavg(f, cfg, runner=runner, on_round=hook)
        time.sleep(2.0)
        log = list(runner.log)
        end = sum(_pid_exists(u["pid"]) for u in log)
    finally:
        runner.close()
    peak = np.array([u["rss_mb"] for u in log])
    num = dict(updates=len(log), distinct_pids=len({u["pid"] for u in log}), peak_rss_mb_mean=float(peak.mean()),
               peak_rss_mb_min=float(peak.min()), peak_rss_mb_max=float(peak.max()),
               alive_right_after_kill=[int(v) for v in now], earlier_round_processes_alive=[int(v) for v in prev],
               seconds_until_round_processes_gone=[float(v) for v in gone], alive_2s_after_run=int(end))
    # k actors alive at the same time, each after one training call
    rows, grow = [], {}
    w0 = init_vector(0).numpy()
    for k in [4, 8, 16, 32]:
        ray.init(num_cpus=k, include_dashboard=False, log_to_driver=False, logging_level="ERROR")
        try:
            t0 = time.time()
            actors = []
            for c in range(k):
                x, y = f.client_train(c)
                actors.append(ClientActor.options(num_cpus=1).remote(c, x.numpy(), y.numpy(), False))
            res = ray.get([a.train.remote(w0, 1, BEST["B"], BEST["lr"], update_seed(0, 1, c)) for c, a in enumerate(actors)])
            secs = time.time() - t0
            pids = [r[2]["pid"] for r in res]
            out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)], capture_output=True, text=True).stdout
            rss = np.array([float(v) for v in out.split()]) / 1024.0  # ps reports kilobytes
            for a in actors:
                ray.kill(a)
        finally:
            ray.shutdown()
        grow[str(k)] = dict(actors_measured=int(len(rss)), total_rss_mb=float(rss.sum()), mean_rss_mb=float(rss.mean()), seconds=secs)
        rows.append([k, len(rss), f"{rss.sum():.0f}", f"{rss.mean():.0f}", f"{secs:.1f}"])
    per_lo, per_hi = min(g["mean_rss_mb"] for g in grow.values()), max(g["mean_rss_mb"] for g in grow.values())
    ram = None
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError):
        pass
    num.update(alive_at_once=grow, estimate_100_actors_gb=[100 * per_lo / 1024, 100 * per_hi / 1024],
               measured_4_actors_gb=grow["4"]["total_rss_mb"] / 1024, machine_ram_gb=ram)
    md = ("The handout states that an actor has a large memory overhead and asks for actors of the selected clients only. "
          f"Part A runs {cfg.rounds} rounds with 4 actors per round and checks, with `os.kill(pid, 0)` on every actor's process id, "
          "which actor processes still exist. Prediction: no process of an earlier round exists when a round ends.\n\n"
          + table(["quantity", "value"], [
              ["client updates / distinct actor processes", f"{num['updates']} / {num['distinct_pids']}"],
              ["peak resident memory of one actor process, MiB (mean, min to max)", f"{peak.mean():.0f} ({peak.min():.0f} to {peak.max():.0f})"],
              ["actor processes of earlier rounds that still exist at the end of a round (max over rounds)", max(prev)],
              ["actor processes of the same round that still exist right after ray.kill (min to max)", f"{min(now)} to {max(now)}"],
              ["seconds until all processes of a round are gone after ray.kill (mean, max)", f"{np.mean(gone):.3f}, {np.max(gone):.3f}"],
              ["actor processes that exist 2 s after the last round", end]])
          + "\nPart B keeps k actors alive at once, each after one training call, and reads their resident memory with `ps`. "
            "Prediction: the total grows in proportion to k, so 100 persistent actors would need 25 times the memory of 4.\n\n"
          + table(["actors alive at once", "processes measured", "total resident memory (MiB)", "mean per actor (MiB)", "seconds to create and train once"], rows)
          + f"\nEstimate for 100 actors alive at once, from the smallest and the largest mean per actor: {100 * per_lo / 1024:.1f} to "
            f"{100 * per_hi / 1024:.1f} GiB, against {grow['4']['total_rss_mb'] / 1024:.2f} GiB measured for 4"
          + (f"; the machine has {ram:.0f} GiB of memory.\n" if ram else ".\n")
          + "Resident memory counts pages of shared libraries in every process, so the sum overstates the unique memory.\n")
    save("ray_memory", "R11. Memory of actor processes and their lifetime", md, num)


# ------------------------------------------------------------------ Part 2 (Laplace noise)
def noise_stats():
    """N1. The injected noise is Laplace(0, b), independent across pixels and of the image."""
    clean = fed(prep="raw")
    rows, num = [], {}
    for seed, b in enumerate([0.1, 1.0, 5.0]):
        noisy = fed(prep="raw", noise_b=b, noise_seed=seed)
        eps = (noisy.x_tr - clean.x_tr).double().numpy()
        flat = eps.reshape(-1)
        sub = np.random.default_rng(0).choice(flat, 200000, replace=False)
        ks = stats.kstest(sub, "laplace", args=(0, b))
        adj = float(np.corrcoef(eps[:, :-1].reshape(-1)[::7], eps[:, 1:].reshape(-1)[::7])[0, 1])
        dep = float(np.corrcoef(flat[::7], clean.x_tr.double().numpy().reshape(-1)[::7])[0, 1])
        d = dict(mean=float(flat.mean()), var_ratio=float(flat.var() / (2 * b * b)), abs_ratio=float(np.abs(flat).mean() / b),
                 tail=float((np.abs(flat) > b).mean()), ks=float(ks.statistic), ks_p=float(ks.pvalue), corr_adjacent=adj, corr_pixel=dep,
                 n=int(flat.size))
        num[str(b)] = d
        rows.append([b, f"{d['mean']:+.5f}", f"{d['var_ratio']:.4f}", f"{d['abs_ratio']:.4f}", f"{d['tail']:.4f}",
                     f"{d['ks']:.4f} (p = {d['ks_p']:.2f})", f"{adj:+.4f}", f"{dep:+.4f}"])
    md = (f"Measured on the noise actually added to the {num['0.1']['n']:,} training pixels (float32 difference of the "
          "perturbed and the clean tensors). Laplace(0, b) has mean 0, variance 2b^2, mean absolute value b and "
          f"P(|e| > b) = 1/e = {np.exp(-1):.4f}. The Kolmogorov-Smirnov test uses 200,000 sampled pixels.\n\n"
          + table(["b", "mean", "variance / 2b^2", "mean |e| / b", "P(|e| > b)", "KS distance to Laplace(0, b)",
                   "correlation of neighbouring pixels' noise", "correlation of noise with the pixel value"], rows))
    save("noise_stats", "N1. Distribution of the injected noise", md, num)


def dp_bound():
    """N2. The privacy loss of the mechanism never exceeds ||x - x'||_1 / b."""
    clients, _ = raw()
    x = torch.from_numpy(np.concatenate([c[0] for c in clients])).double()
    g = torch.Generator().manual_seed(0)
    n_pairs = 20000
    i, j = torch.randint(0, len(x), (n_pairs,), generator=g), torch.randint(0, len(x), (n_pairs,), generator=g)
    keep = i != j
    xi, xj = x[i[keep]], x[j[keep]]
    l1 = (xi - xj).abs().sum(1)
    num = dict(pairs=int(keep.sum()), l1_mean=float(l1.mean()), l1_max=float(l1.max()), l1_min=float(l1.min()),
               ink_mean=float((1 - x).sum(1).mean()), ink_max=float((1 - x).sum(1).max()))
    rows = []
    rng = np.random.default_rng(0)
    for b in [0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0]:
        o = xi + torch.from_numpy(rng.laplace(0, b, size=tuple(xi.shape)))
        loss = ((o - xj).abs().sum(1) - (o - xi).abs().sum(1)) / b      # log p(o | x) - log p(o | x')
        bound = l1 / b
        d = dict(max_ratio=float((loss / bound).max()), violations=int((loss > bound + 1e-9).sum()),
                 mean_loss=float(loss.mean()), wrong_guess=float((loss < 0).double().mean()),
                 eps_image_worst=784 / b, eps_pixel=1 / b, eps_typical=float(l1.mean() / b))
        num[str(b)] = d
        rows.append([b, f"{1 / b:.2f}", f"{784 / b:.0f}", f"{d['eps_typical']:.1f}", f"{d['mean_loss']:.2f}",
                     f"{d['max_ratio']:.3f}", d["violations"], f"{d['wrong_guess']:.4f}"])
    md = ("For a released image o = x + e the log-likelihood ratio between two candidate originals x and x' is "
          "(||o - x'||_1 - ||o - x||_1) / b. The triangle inequality bounds it by ||x - x'||_1 / b, which is the "
          "epsilon of the Laplace mechanism for that pair; the worst case over all images in [0, 1]^784 is 784 / b. "
          f"Checked on {num['pairs']:,} random pairs of training images: their L1 distance is {num['l1_mean']:.1f} on "
          f"average and at most {num['l1_max']:.1f}.\n\n"
          + table(["b", "epsilon per pixel (1/b)", "epsilon per image, worst case (784/b)", "epsilon for an average pair",
                   "mean measured privacy loss", "largest loss / bound", "pairs above the bound",
                   "share of pairs where the released image favours the wrong original"], rows)
          + "\nThe last column is the error rate of the best possible guess between two equally likely candidate images.\n")
    save("dp_bound", "N2. The differential-privacy inequality of the Laplace mechanism", md, num)


def reident():
    """N4. What an observer of the perturbed images can still do."""
    clients, (xt_raw, yt) = raw()
    x = torch.from_numpy(np.concatenate([c[0] for c in clients])).double()
    y = torch.from_numpy(np.concatenate([c[1] for c in clients]))
    g = torch.Generator().manual_seed(1)
    cand = torch.randperm(len(x), generator=g)[:2000]
    xc, yc = x[cand], y[cand]
    targets = torch.arange(500)
    f = fed()
    w = torch.load(os.path.join(RES, "main", "seed0", "final_weights.pt"))
    xt = torch.from_numpy(xt_raw).double()
    rng = np.random.default_rng(2)
    rows, num = [], {}
    scales = [b for b in NOISE_SCALES if b > 0] + [10.0, 20.0, 50.0]
    for b in scales:
        o = xc[targets] + torch.from_numpy(rng.laplace(0, b, size=(len(targets), 784)))
        d = torch.cdist(o, xc, p=1)
        guess = d.argmin(1)
        ident = float((guess == targets).double().mean())
        label = float((yc[guess] == yc[targets]).double().mean())
        noisy_test = (1.0 - (xt + torch.from_numpy(rng.laplace(0, b, size=tuple(xt.shape))))).float()
        acc = float((forward(w, noisy_test).argmax(1) == f.y_te).double().mean())
        snr = float(((1 - xc) ** 2).sum(1).mean() / (784 * 2 * b * b))
        num[str(b)] = dict(ident=ident, label=label, clean_model_acc=acc, snr=snr)
        rows.append([b, f"{784 / b:.0f}", f"{snr:.3f}", f"{ident:.3f}", f"{label:.3f}", f"{acc:.4f}"])
    num["chance_ident"] = 1 / 2000
    md = ("An observer holds 2,000 candidate clean images and one perturbed image of one of them, and names the "
          "candidate with the smallest L1 distance, which is the maximum-likelihood guess under Laplace noise. Chance "
          "is 0.0005. 'Label read' is the share of guesses with the correct class. The last column feeds perturbed "
          "test images to the model trained on clean data (seed 0). The energy ratio is the mean squared ink "
          "of an image divided by the noise energy 784 x 2b^2.\n\n"
          + table(["b", "epsilon per image, worst case", "ink energy / noise energy", "original identified among 2,000",
                   "label read from the nearest candidate", "clean-trained model on perturbed test images"], rows))
    save("reident", "N4. Re-identification of perturbed images", md, num)


def noise_checks():
    """N3, N5. b = 0 reproduces Part 1; summary of the noise sweep and its edge-case variants."""
    num, md = {}, ""
    a, b = os.path.join(RES, "noise", "b0.0_seed0", "final_weights.pt"), os.path.join(RES, "main", "seed0", "final_weights.pt")
    if os.path.exists(a) and os.path.exists(b):
        num["b0_equal"] = bool(torch.equal(torch.load(a), torch.load(b)))
        md += (f"**N3.** The run with b = 0 (`results/noise/b0.0_seed0`) and the Part 1 run (`results/main/seed0`) have "
               f"identical final weights: {num['b0_equal']}.\n\n")
    rows = []
    f0 = fed()
    ink = float((f0.x_tr.double() ** 2).sum(1).mean())
    num["ink_energy"] = ink
    for bb in NOISE_SCALES:
        paths = sorted(glob.glob(os.path.join(RES, "noise", f"b{bb}_seed*")))
        runs = [load_run(p) for p in paths]
        if not runs:
            continue
        col = lambda c: [float(h[c].iloc[-1]) for h, _, _ in runs]
        classes, unstable, frozen = [], 0, []
        for p, (h, _, _) in zip(paths, runs):
            w = torch.load(os.path.join(p, "final_weights.pt"))
            classes.append(int(len(torch.unique(forward(w, f0.x_te).argmax(1)))))
            unstable += int((~np.isfinite(h.train_loss)).any() or h.train_loss.max() > 20)
            frozen.append(int((h.n_returned.iloc[1:] == 0).sum()))
        d = dict(train=col("train_acc"), val=col("val_acc"), test=[s["test_acc"] for _, s, _ in runs],
                 clean_train=col("clean_train_acc") if bb > 0 else col("train_acc"), train_loss=col("train_loss"),
                 classes_predicted=classes, seeds_unstable=unstable, rounds_without_update=frozen,
                 input_energy=ink + 784 * 2 * bb * bb)
        num[str(bb)] = d
        rows.append([bb, f"{d['input_energy']:.0f}", f"{np.mean(d['train']):.4f}", f"{np.mean(d['clean_train']):.4f}",
                     f"{np.mean(d['val']):.4f}", f"{np.mean(d['test']):.4f}", f"{np.std(d['test'], ddof=1):.4f}",
                     f"{unstable} of {len(runs)}", "/".join(str(v) for v in frozen), "/".join(str(c) for c in classes)])
    if rows:
        md += ("**N5.** Round-300 accuracies at learning rate 0.5, mean of 3 seeds. 'Training (perturbed)' is measured on "
               "the images the clients trained on; 'training (clean)' on the same images before the noise. The input "
               f"energy is the mean squared norm of one network input, {ink:.1f} + 1568 b^2. A seed counts as unstable "
               "when its aggregated training loss exceeded 20 or left the finite range in some round. 'Rounds without "
               "update' counts rounds in which every selected client returned non-finite weights, so the server kept "
               "the old model.\n\n"
               + table(["b", "input energy", "training (perturbed)", "training (clean)", "validation (perturbed)", "test (clean)",
                        "std of test over seeds", "unstable seeds", "rounds without update, per seed",
                        "distinct classes predicted on the test set, per seed"], rows))
    rows, same = [], []
    for bb in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        r = [bb]
        for name in [f"b{bb}_seed0", f"clip_b{bb}", f"cleanval_b{bb}"]:
            folder = "noise" if name.startswith("b") else "noise_variants"
            p = os.path.join(RES, folder, name)
            if os.path.exists(p):
                h, s, _ = load_run(p)
                num[name] = dict(train=float(h.train_acc.iloc[-1]), val=float(h.val_acc.iloc[-1]), test=s["test_acc"])
                r += [f"{h.train_acc.iloc[-1]:.4f}", f"{h.val_acc.iloc[-1]:.4f}", f"{s['test_acc']:.4f}"]
            else:
                r += ["", "", ""]
        rows.append(r)
        pa = os.path.join(RES, "noise", f"b{bb}_seed0", "final_weights.pt")
        pb = os.path.join(RES, "noise_variants", f"cleanval_b{bb}", "final_weights.pt")
        if os.path.exists(pa) and os.path.exists(pb):
            same.append(bool(torch.equal(torch.load(pa), torch.load(pb))))
    num["cleanval_weights_identical"] = same
    md += ("\n**N6. Edge-case variants (seed 0).** 'Clipped' clips perturbed pixels to [0, 1] before training. 'Clean "
           "validation' perturbs the training split only.\n\n"
           + table(["b", "train", "validation", "test", "clipped: train", "clipped: validation", "clipped: test",
                    "clean validation: train", "clean validation: validation", "clean validation: test"], rows)
           + f"\nFinal weights of the clean-validation run identical to the standard run (torch.equal): "
             f"{sum(same)} of {len(same)} scales.\n")
    if glob.glob(os.path.join(RES, "noise_lr", "*")):
        from .make_plots import noise_lr_selected, noise_lr_table
        d, sel = noise_lr_table(), noise_lr_selected()
        rows = []
        for _, r in d.iterrows():
            chosen = bool(r.E == 5 and r.b in sel.index and sel.loc[r.b, "lr"] == r.lr)
            state = "diverged" if r.diverged else ("one class on clean images" if r.clean < 0.04 and r.train > 0.1 else "")
            num[f"lr_b{r.b}_lr{r.lr}_E{r.E}"] = dict(train=float(r.train), val=float(r.val), clean=float(r.clean), test=float(r.test),
                                                     diverged=bool(r.diverged), chosen=chosen, product=float(r.lr * r.b ** 2),
                                                     best_val_round=int(r.best_val_round), clean_at_best_val=float(r.clean_at_best_val))
            rows.append([r.b, r.lr, int(r.E), f"{r.lr * r.b ** 2:.4f}", f"{r.train:.4f}", f"{r.val:.4f}", f"{r.clean:.4f}", f"{r.test:.4f}",
                         f"{int(r.best_val_round)} / {r.clean_at_best_val:.4f}", state, "chosen" if chosen else ""])
        md += ("\n**N7. Learning rate re-tuned per noise scale (seed 0, 300 rounds).** 'Chosen' marks the E = 5 run with the "
               "highest round-300 validation accuracy among the runs that did not diverge; the test column was not used "
               "for the choice. The column 'best round' gives the round with the highest validation accuracy and the "
               "clean training accuracy in that round, which is what early stopping on the perturbed validation split "
               "would return.\n\n"
               + table(["b", "learning rate", "E", "learning rate x b^2", "training (perturbed)", "validation (perturbed)",
                        "training (clean)", "test (clean)", "best round / clean training accuracy there", "state", "choice"], rows))
    save("noise_checks", "N3, N5, N6, N7. Noise sweep checks", md, num)


SECTIONS = dict(fedsgd=fedsgd, trace=trace, references=references, sampling=sampling, structure=structure,
                errors=errors, drift=drift, ray_equal=ray_equal, ray_faults=ray_faults, ray_scaling=ray_scaling,
                ray_memory=ray_memory, noise_stats=noise_stats, dp_bound=dp_bound, reident=reident, noise_checks=noise_checks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        for k, fn in SECTIONS.items():
            print(f"{k:14s} {fn.__doc__}")
        return
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    for name in (a.only or SECTIONS):
        SECTIONS[name]()
    with open(os.path.join(OUT, "validation.md"), "w") as out:
        out.write("# Validation experiments\n\nWritten by `python -m fedavg.validate`.\n\n")
        for name in list(SECTIONS) + ["gpu_checks"]:  # gpu_checks.md is written by python -m fedavg.gpu_checks --report
            path = os.path.join(OUT, name + ".md")
            if os.path.exists(path):
                out.write(open(path).read() + "\n")


if __name__ == "__main__":
    main()
