"""Run a named group of FedAvg experiments in a small process pool.

    python -m fedavg.sweep --name tune            # learning rate and batch size, C = 10%, E = 5
    python -m fedavg.sweep --name ce              # C x E comparison, 3 seeds
    python -m fedavg.sweep --name main            # chosen hyperparameters, 5 seeds
    python -m fedavg.sweep --name noise           # Part 2: Laplace scale b, 3 seeds
    python -m fedavg.sweep --name noise_lr        # Part 2: learning rate re-tuned per noise scale

Every run writes results/<name>/<run>/{history.csv, arrays.npz, summary.json, final_weights.pt}.
A run whose summary.json exists is skipped. The weight files are not in the repository (81 MB);
--restore-weights retrains the runs whose summary exists but whose weight file is missing and
writes only final_weights.pt (the stored history is kept).
"""
import argparse
import itertools
import os
import time
from multiprocessing import get_context

# hyperparameters chosen from the 'tune' and 'ce' sweeps on validation accuracy (see the report)
BEST = dict(E=5, B=20, lr=0.5, rounds=300)
NOISE_SCALES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def grid(name):
    runs = []
    if name == "tune":
        for lr, B in itertools.product([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0], [10, 20, 50]):
            runs.append((f"lr{lr}_B{B}", dict(C=0.1, E=5, B=B, lr=lr, rounds=300, seed=0)))
    elif name == "ce":
        for C, E, seed in itertools.product([0.01, 0.02, 0.05, 0.1], [1, 2, 5, 10, 20], [0, 1, 2]):
            runs.append((f"C{C}_E{E}_seed{seed}", dict(C=C, E=E, B=BEST["B"], lr=BEST["lr"], rounds=300, seed=seed)))
    elif name == "main":
        for seed in range(5):
            runs.append((f"seed{seed}", dict(C=0.1, seed=seed, **BEST)))
    elif name == "noise":
        for b, seed in itertools.product(NOISE_SCALES, [0, 1, 2]):
            runs.append((f"b{b}_seed{seed}", dict(C=0.1, seed=seed, noise_b=b, noise_seed=seed, **BEST)))
    elif name == "noise_variants":
        # edge cases of the noise sweep: pixels clipped back to [0, 1]; validation split left clean
        for b in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
            runs.append((f"clip_b{b}", dict(C=0.1, seed=0, noise_b=b, noise_seed=0, clip=True, **BEST)))
            runs.append((f"cleanval_b{b}", dict(C=0.1, seed=0, noise_b=b, noise_seed=0, noise_val=False, **BEST)))
    elif name == "noise_lr":
        # the learning rate tuned on clean images (0.5) is re-tuned for perturbed images; E = 1 as a second remedy
        for b, lr in itertools.product([0.2, 0.5, 1.0, 2.0, 5.0], [0.1, 0.02, 0.005, 0.001]):
            runs.append((f"b{b}_lr{lr}", dict(C=0.1, seed=0, noise_b=b, noise_seed=0, **{**BEST, "lr": lr})))
        for b, lr in itertools.product([0.2, 0.5, 1.0], [0.5, 0.1]):
            runs.append((f"b{b}_lr{lr}_E1", dict(C=0.1, seed=0, noise_b=b, noise_seed=0, **{**BEST, "lr": lr, "E": 1})))
    elif name == "checks":
        # validation experiments that change the structure of the problem (see fedavg/validate.py)
        for part, E, seed in itertools.product(["writer", "iid"], [1, 5, 20], [0, 1]):
            runs.append((f"{part}_E{E}_seed{seed}", dict(C=0.1, seed=seed, partition=part, **{**BEST, "E": E})))
        for prep in ["invert", "raw", "center"]:
            for lr in [0.05, 0.5]:
                runs.append((f"prep_{prep}_lr{lr}", dict(C=0.1, seed=0, prep=prep, **{**BEST, "lr": lr})))
        for seed in range(3):
            runs.append((f"uniform_seed{seed}", dict(C=0.1, seed=seed, weighting="uniform", **BEST)))
        runs.append(("shuffled_labels", dict(C=0.1, seed=0, shuffle_labels=True, **BEST)))
        for decay in [0.995, 0.99]:
            runs.append((f"lrdecay{decay}", dict(C=0.1, seed=0, lr_decay=decay, **BEST)))
    else:
        raise SystemExit(f"unknown sweep {name}")
    return runs


_RAW = None


def run_one(job):
    out, kw, restore = job
    have_summary = os.path.exists(os.path.join(out, "summary.json"))
    if have_summary and (not restore or os.path.exists(os.path.join(out, "final_weights.pt"))):
        return out, None
    import torch
    from .core import Config, clients_from_fraction, run_fedavg, save_run
    from .data import build_federation, load_raw
    global _RAW
    torch.set_num_threads(1)
    if _RAW is None:
        _RAW = load_raw(kw.get("data"))
    fed = build_federation(raw=_RAW, split_seed=kw.get("split_seed", 0), noise_b=kw.get("noise_b", 0.0),
                           noise_seed=kw.get("noise_seed", 0), noise_val=kw.get("noise_val", True),
                           clip=kw.get("clip", False), prep=kw.get("prep", "invert"),
                           partition=kw.get("partition", "writer"), shuffle_labels=kw.get("shuffle_labels", False))
    m = kw.get("clients_per_round") or clients_from_fraction(kw["C"], fed.n_clients)
    cfg = Config(rounds=kw["rounds"], clients_per_round=m, epochs=kw["E"], batch_size=kw["B"], lr=kw["lr"],
                 seed=kw["seed"], weighting=kw.get("weighting", "samples"), lr_decay=kw.get("lr_decay", 1.0))
    t0 = time.time()
    w, hist = run_fedavg(fed, cfg)
    if have_summary:  # only the weight file was missing
        torch.save(w, os.path.join(out, "final_weights.pt"))
        return out, "restored"
    s = save_run(out, cfg, fed, w, hist, extra=dict(wall_seconds=time.time() - t0, ray=False))
    return out, s


def run_jobs(jobs, workers):
    with get_context("spawn").Pool(workers) as pool:
        for out, s in pool.imap_unordered(run_one, jobs):
            if s is None:
                print(f"skip  {out}", flush=True)
            elif s == "restored":
                print(f"restored weights  {out}", flush=True)
            else:
                print(f"done  {out}  train {s['final']['train_acc']:.4f}  val {s['final']['val_acc']:.4f}  "
                      f"test {s['test_acc']:.4f}  {s['wall_seconds']:.0f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--restore-weights", action="store_true")
    a = p.parse_args()
    jobs = [(os.path.join(a.results, a.name, run), kw, a.restore_weights) for run, kw in grid(a.name)]
    run_jobs(jobs, a.workers)


if __name__ == "__main__":
    main()
