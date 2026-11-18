"""FedAvg (McMahan et al. 2017) written with PyTorch tensors only.

The model is the required 2-layer fully connected network 784 -> 128 (ReLU) -> 62. Its
108,478 parameters are kept as one flat vector, so the server step is one weighted sum.
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import N_CLASSES, N_PIXELS

HIDDEN = 128
SHAPES = [(HIDDEN, N_PIXELS), (HIDDEN,), (N_CLASSES, HIDDEN), (N_CLASSES,)]
SIZES = [int(np.prod(s)) for s in SHAPES]
N_PARAMS = sum(SIZES)  # 108,478


class MLP(torch.nn.Module):
    """The same network as an nn.Module; used to check the flat-vector forward pass."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(N_PIXELS, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    def load_vector(self, vec):
        w1, b1, w2, b2 = unpack(vec)
        with torch.no_grad():
            self.fc1.weight.copy_(w1); self.fc1.bias.copy_(b1)
            self.fc2.weight.copy_(w2); self.fc2.bias.copy_(b2)
        return self


def unpack(vec):
    out, i = [], 0
    for shape, size in zip(SHAPES, SIZES):
        out.append(vec[i:i + size].view(shape))
        i += size
    return out


def forward(vec, x):
    w1, b1, w2, b2 = unpack(vec)
    return F.linear(torch.relu(F.linear(x, w1, b1)), w2, b2)


def init_vector(seed, dtype=torch.float32):
    """PyTorch's default nn.Linear initialisation, U(-1/sqrt(fan_in), 1/sqrt(fan_in)), from a private generator."""
    g = torch.Generator().manual_seed(int(seed))
    parts = []
    for shape, fan_in in zip(SHAPES, [N_PIXELS, N_PIXELS, HIDDEN, HIDDEN]):
        bound = 1.0 / math.sqrt(fan_in)
        parts.append(((torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * bound).reshape(-1))
    return torch.cat(parts).to(dtype)


def update_seed(seed, rnd, client_id):
    """Seed of one client update. It depends on (run seed, round, client) only, so the result of a
    client update does not depend on which process runs it or in which order."""
    return int(np.random.SeedSequence([int(seed), int(rnd), int(client_id)]).generate_state(1)[0])


def client_update(global_vec, x, y, epochs, batch_size, lr, seed):
    """E epochs of minibatch SGD on one client's training data, started from the global weights.

    batch_size <= 0 or >= n means one full-batch step per epoch. The last, smaller batch of an
    epoch is used. Returns the new weight vector and a small statistics dictionary.
    """
    n = int(len(y))
    stats = dict(n=n, steps=0, loss_first=float("nan"), loss_last=float("nan"), finite=True)
    if n == 0 or epochs <= 0:
        return global_vec.clone(), stats
    w = global_vec.clone().requires_grad_(True)
    bs = n if (batch_size is None or batch_size <= 0 or batch_size >= n) else int(batch_size)
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(epochs)):
        # the order is drawn on the CPU, so it is the same on every device; the index then follows the data
        perm = torch.randperm(n, generator=g).to(x.device) if bs < n else None
        loss_sum = 0.0
        for i in range(0, n, bs):
            if perm is None:
                xb, yb = x, y
            else:
                idx = perm[i:i + bs]
                xb, yb = x[idx], y[idx]
            loss = F.cross_entropy(forward(w, xb), yb)
            (grad,) = torch.autograd.grad(loss, w)
            with torch.no_grad():
                w -= lr * grad
            lv = loss.item()
            if stats["steps"] == 0:
                stats["loss_first"] = lv
            loss_sum += lv * len(yb)
            stats["steps"] += 1
        stats["loss_last"] = loss_sum / n
    w = w.detach()
    stats["finite"] = bool(torch.isfinite(w).all())
    return w, stats


def aggregate(vectors, weights):
    """Server step: sum_k p_k w_k with p_k = weights_k / sum(weights)."""
    p = torch.as_tensor(np.asarray(weights, dtype=np.float64))
    if len(vectors) == 0 or float(p.sum()) <= 0:
        raise ValueError("nothing to aggregate")
    if (p < 0).any():
        raise ValueError("negative aggregation weight")
    p = (p / p.sum()).to(vectors[0].dtype)
    return torch.stack(vectors, dim=1) @ p


@torch.no_grad()
def evaluate(vec, x, y, client_index=None, n_clients=None):
    """Mean cross-entropy and accuracy; with client_index also the per-client sums."""
    logits = forward(vec, x)
    losses = F.cross_entropy(logits, y, reduction="none").double()
    correct = (logits.argmax(1) == y).double()
    out = dict(loss=float(losses.mean()), acc=float(correct.mean()), n=int(len(y)))
    if client_index is not None:
        out["client_loss_sum"] = torch.bincount(client_index, weights=losses, minlength=n_clients).numpy()
        out["client_correct"] = torch.bincount(client_index, weights=correct, minlength=n_clients).numpy()
    return out


@dataclass
class Config:
    rounds: int = 200
    clients_per_round: int = 10   # m = C * K
    epochs: int = 5               # E
    batch_size: int = 20          # B (0 = full batch)
    lr: float = 0.05
    seed: int = 0
    weighting: str = "samples"    # 'samples' (n_k / n) or 'uniform' (1 / m)
    lr_decay: float = 1.0         # multiplicative decay of the client learning rate per round


def clients_from_fraction(c_fraction, n_clients):
    """m = max(1, round(C K)), capped at K."""
    return int(min(n_clients, max(1, round(c_fraction * n_clients))))


class SequentialRunner:
    """Runs the selected clients one after another in this process."""

    def __init__(self, fed, cfg):
        self.fed, self.cfg = fed, cfg

    def __call__(self, rnd, selected, global_vec, lr):
        out = []
        for k in selected:
            x, y = self.fed.client_train(int(k))
            t0 = time.time()
            vec, stats = client_update(global_vec, x, y, self.cfg.epochs, self.cfg.batch_size, lr,
                                       update_seed(self.cfg.seed, rnd, k))
            stats.update(t_start=t0, t_end=time.time(), pid=os.getpid())
            out.append(dict(cid=int(k), vec=vec, stats=stats))
        return out

    def close(self):
        pass


def run_fedavg(fed, cfg, runner=None, on_round=None, verbose=False, init_vec=None):
    """Train with FedAvg. After every round the new global model is evaluated on the training
    and validation data of all clients. Returns (final weights, history dictionary)."""
    K = fed.n_clients
    m = int(cfg.clients_per_round)
    if not 1 <= m <= K:
        raise ValueError(f"clients_per_round must be in 1..{K}")
    own_runner = runner is None
    runner = runner or SequentialRunner(fed, cfg)
    w = init_vector(cfg.seed, fed.x_tr.dtype) if init_vec is None else init_vec.clone()
    rng = np.random.default_rng([cfg.seed, 11])
    idx_tr, idx_va = fed.client_index("train"), fed.client_index("val")
    n_tr, n_va = fed.n_tr.astype(float), fed.n_va.astype(float)

    cols = ["round", "train_loss", "train_acc", "val_loss", "val_acc", "sel_train_acc", "sel_val_acc",
            "local_loss_first", "local_loss_last", "drift", "local_norm", "update_norm", "n_returned", "n_selected",
            "samples_processed", "seconds", "clean_train_acc", "clean_train_loss"]
    hist = {c: [] for c in cols}
    client_val_acc = np.zeros((cfg.rounds + 1, K))
    client_train_acc = np.zeros((cfg.rounds + 1, K))
    selected_mask = np.zeros((cfg.rounds + 1, K), dtype=bool)
    timings = []
    samples = 0

    def record(rnd, sel, results, w_old, seconds):
        tr = evaluate(w, fed.x_tr, fed.y_tr, idx_tr, K)
        va = evaluate(w, fed.x_va, fed.y_va, idx_va, K)
        # a client without images has no accuracy: NaN, and it does not enter the pooled means above
        client_train_acc[rnd] = np.where(n_tr > 0, tr["client_correct"] / np.maximum(n_tr, 1), np.nan)
        client_val_acc[rnd] = np.where(n_va > 0, va["client_correct"] / np.maximum(n_va, 1), np.nan)
        row = dict(round=rnd, train_loss=tr["loss"], train_acc=tr["acc"], val_loss=va["loss"], val_acc=va["acc"],
                   n_selected=len(sel), n_returned=len(results), samples_processed=samples, seconds=seconds)
        if len(sel):
            row["sel_train_acc"] = tr["client_correct"][sel].sum() / max(n_tr[sel].sum(), 1)
            row["sel_val_acc"] = va["client_correct"][sel].sum() / max(n_va[sel].sum(), 1)
        if results:
            ns = np.array([r["stats"]["n"] for r in results], dtype=float)
            p = ns / ns.sum()
            row["local_loss_first"] = float(np.dot(p, [r["stats"]["loss_first"] for r in results]))
            row["local_loss_last"] = float(np.dot(p, [r["stats"]["loss_last"] for r in results]))
            row["drift"] = float(np.dot(p, [float((r["vec"] - w).norm()) for r in results]))
            # mean distance the clients moved, and the distance the server moved
            row["local_norm"] = float(np.dot(p, [float((r["vec"] - w_old).norm()) for r in results]))
            row["update_norm"] = float((w - w_old).norm())
        if fed.x_tr_clean is not None:
            cl = evaluate(w, fed.x_tr_clean, fed.y_tr)
            row["clean_train_acc"], row["clean_train_loss"] = cl["acc"], cl["loss"]
        for c in cols:
            hist[c].append(row.get(c, float("nan")))
        if verbose and (rnd % verbose == 0 or rnd == cfg.rounds):
            print(f"round {rnd:4d}  train loss {tr['loss']:.4f} acc {tr['acc']:.4f}  val loss {va['loss']:.4f} "
                  f"acc {va['acc']:.4f}  returned {len(results)}/{len(sel)}", flush=True)

    record(0, np.array([], dtype=int), [], w, 0.0)
    try:
        for rnd in range(1, cfg.rounds + 1):
            t0 = time.time()
            sel = np.sort(rng.choice(K, size=m, replace=False))
            selected_mask[rnd, sel] = True
            lr = cfg.lr * cfg.lr_decay ** (rnd - 1)
            results = runner(rnd, sel, w, lr)
            # a client that failed, timed out, has no data or returned non-finite weights is left out
            results = [r for r in results if r is not None and r["stats"]["n"] > 0 and r["stats"]["finite"]]
            w_old = w
            if results:
                ns = [r["stats"]["n"] for r in results]
                weights = ns if cfg.weighting == "samples" else [1.0] * len(ns)
                w = aggregate([r["vec"] for r in results], weights)
                samples += int(sum(r["stats"]["n"] for r in results)) * cfg.epochs
            for r in results:
                timings.append((rnd, r["cid"], r["stats"].get("pid", -1), r["stats"].get("t_start", np.nan),
                                r["stats"].get("t_end", np.nan), r["stats"].get("t_created", np.nan)))
            if on_round is not None:
                on_round(rnd, w_old, results, w)
            record(rnd, sel, results, w_old, time.time() - t0)
    finally:
        if own_runner:
            runner.close()
    history = {c: np.array(v, dtype=float) for c, v in hist.items()}
    history.update(client_val_acc=client_val_acc, client_train_acc=client_train_acc,
                   selected=selected_mask, timings=np.array(timings, dtype=float).reshape(-1, 6))
    return w, history


def test_metrics(w, fed):
    te = evaluate(w, fed.x_te, fed.y_te)
    return dict(test_loss=te["loss"], test_acc=te["acc"], test_n=te["n"])


def save_run(out, cfg, fed, w, history, extra=None):
    os.makedirs(out, exist_ok=True)
    scalar = {k: v for k, v in history.items() if v.ndim == 1}
    import pandas as pd
    pd.DataFrame(scalar).to_csv(os.path.join(out, "history.csv"), index=False)
    np.savez_compressed(os.path.join(out, "arrays.npz"), client_val_acc=history["client_val_acc"],
                        client_train_acc=history["client_train_acc"], selected=history["selected"],
                        timings=history["timings"])
    torch.save(w, os.path.join(out, "final_weights.pt"))
    summary = dict(config=asdict(cfg), data=fed.meta, n_clients=fed.n_clients,
                   final={k: float(v[-1]) for k, v in scalar.items()}, **test_metrics(w, fed))
    summary.update(extra or {})
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
