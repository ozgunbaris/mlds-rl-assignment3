"""Train one FedAvg model and save its history.

    python -m fedavg.run --out results/try --C 0.1 --E 5 --seed 0
    python -m fedavg.run --out results/try_ray --clients-per-round 4 --ray
    python -m fedavg.run --out results/try_noise --C 0.1 --noise-b 0.5

The defaults are the hyperparameters chosen in the report: E = 5, B = 20, learning rate 0.5, 300 rounds.
"""
import argparse
import time

import torch

from .core import Config, clients_from_fraction, run_fedavg, save_run
from .data import build_federation


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None, help="folder with train_data.npy and test_data.npy (default: data/)")
    p.add_argument("--C", type=float, default=0.1, help="fraction of clients per round")
    p.add_argument("--clients-per-round", type=int, default=None, help="overrides --C")
    p.add_argument("--E", type=int, default=5, help="local epochs")
    p.add_argument("--B", type=int, default=20, help="local batch size (0 = full batch)")
    p.add_argument("--lr", type=float, default=0.5, help="client learning rate")
    p.add_argument("--lr-decay", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--weighting", choices=["samples", "uniform"], default="samples")
    p.add_argument("--prep", choices=["invert", "raw", "center"], default="invert")
    p.add_argument("--partition", choices=["writer", "iid"], default="writer")
    p.add_argument("--shuffle-labels", action="store_true")
    p.add_argument("--noise-b", type=float, default=0.0, help="Laplace scale b on the [0, 1] pixel scale")
    p.add_argument("--noise-seed", type=int, default=None, help="seed of the Laplace noise (default: --seed, as in the sweeps)")
    p.add_argument("--noise-train-only", action="store_true", help="leave the validation split clean")
    p.add_argument("--clip", action="store_true", help="clip perturbed pixels to [0, 1]")
    p.add_argument("--ray", action="store_true", help="run the client updates as Ray actors")
    p.add_argument("--ray-cpus", type=int, default=None)
    p.add_argument("--ray-timeout", type=float, default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--verbose", type=int, default=10)
    return p


def main(argv=None):
    a = add_args(argparse.ArgumentParser(description=__doc__)).parse_args(argv)
    torch.set_num_threads(a.threads)
    noise_seed = a.seed if a.noise_seed is None else a.noise_seed
    fed = build_federation(a.data, split_seed=a.split_seed, noise_b=a.noise_b, noise_seed=noise_seed,
                           noise_val=not a.noise_train_only, clip=a.clip, prep=a.prep,
                           partition=a.partition, shuffle_labels=a.shuffle_labels)
    m = a.clients_per_round or clients_from_fraction(a.C, fed.n_clients)
    cfg = Config(rounds=a.rounds, clients_per_round=m, epochs=a.E, batch_size=a.B, lr=a.lr, seed=a.seed,
                 weighting=a.weighting, lr_decay=a.lr_decay)
    runner = None
    if a.ray:
        from .ray_clients import RayRunner
        runner = RayRunner(fed, cfg, num_cpus=a.ray_cpus, timeout=a.ray_timeout)
    t0 = time.time()
    try:
        w, hist = run_fedavg(fed, cfg, runner=runner, verbose=a.verbose)
    finally:
        if runner is not None:
            runner.close()
    s = save_run(a.out, cfg, fed, w, hist, extra=dict(wall_seconds=time.time() - t0, ray=a.ray))
    print(f"final: train acc {s['final']['train_acc']:.4f}  val acc {s['final']['val_acc']:.4f}  "
          f"test acc {s['test_acc']:.4f}  ({s['wall_seconds']:.1f} s)  -> {a.out}")


if __name__ == "__main__":
    main()
