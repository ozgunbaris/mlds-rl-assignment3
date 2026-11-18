"""Mechanical checks of the FedAvg code. Run with: python -m pytest tests -q

The checks use small synthetic federations in float64, so equalities hold to about 1e-12.
The experiments that test the behaviour on the real data are in fedavg/validate.py.
"""
import os

import numpy as np
import pytest
import torch

from fedavg import core, data
from fedavg.core import Config, aggregate, client_update, forward, init_vector, run_fedavg

D64 = torch.float64


def make_fed(sizes, seed=0, dtype=D64, val=4):
    """Random federation: client k has sizes[k] training rows and `val` validation rows."""
    g = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    off = np.concatenate([[0], np.cumsum(sizes)])
    off_va = np.arange(len(sizes) + 1) * val
    nv = len(sizes) * val
    return data.Federation(
        x_tr=torch.rand(n, 784, generator=g, dtype=dtype), y_tr=torch.randint(0, 62, (n,), generator=g),
        x_va=torch.rand(nv, 784, generator=g, dtype=dtype), y_va=torch.randint(0, 62, (nv,), generator=g),
        off_tr=off, off_va=off_va,
        x_te=torch.rand(8, 784, generator=g, dtype=dtype), y_te=torch.randint(0, 62, (8,), generator=g))


def test_parameter_count_and_forward_matches_module():
    assert core.N_PARAMS == 784 * 128 + 128 + 128 * 62 + 62 == 108478
    w = init_vector(3, D64)
    x = torch.rand(5, 784, dtype=D64)
    module = core.MLP().double().load_vector(w)
    assert torch.allclose(forward(w, x), module(x), atol=1e-14)
    assert sum(p.numel() for p in module.parameters()) == core.N_PARAMS


def test_init_is_seeded_and_bounded():
    a, b, c = init_vector(0), init_vector(0), init_vector(1)
    assert torch.equal(a, b) and not torch.equal(a, c)
    w1, b1, w2, b2 = core.unpack(a)
    assert w1.abs().max() <= 1 / 28 and w2.abs().max() <= 1 / np.sqrt(128)


def test_aggregate_is_the_weighted_mean():
    v = [torch.tensor([1.0, 0.0], dtype=D64), torch.tensor([0.0, 1.0], dtype=D64), torch.tensor([4.0, 4.0], dtype=D64)]
    out = aggregate(v, [1, 3, 4])
    assert torch.allclose(out, torch.tensor([1 / 8 + 2.0, 3 / 8 + 2.0], dtype=D64))
    assert torch.allclose(aggregate(v, [2, 6, 8]), out)                 # only the ratios matter
    assert torch.allclose(aggregate(v[::-1], [4, 3, 1]), out)           # order does not matter
    assert torch.allclose(aggregate([v[2]] * 5, [3, 1, 4, 1, 5]), v[2])  # identical models stay unchanged
    assert torch.allclose(aggregate(v, [0, 0, 7]), v[2])                # a zero-weight client has no effect
    with pytest.raises(ValueError):
        aggregate(v, [0, 0, 0])
    with pytest.raises(ValueError):
        aggregate([], [])


def test_client_update_matches_torch_sgd():
    """E epochs of minibatch SGD must equal torch.optim.SGD run on an nn.Module with the same batches."""
    fed = make_fed([37])
    x, y = fed.client_train(0)
    w0 = init_vector(0, D64)
    w, stats = client_update(w0, x, y, epochs=3, batch_size=8, lr=0.1, seed=5)
    module = core.MLP().double().load_vector(w0)
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    g = torch.Generator().manual_seed(5)
    for _ in range(3):
        perm = torch.randperm(37, generator=g)
        for i in range(0, 37, 8):
            idx = perm[i:i + 8]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(module(x[idx]), y[idx]).backward()
            opt.step()
    ref = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    assert torch.allclose(w, ref, atol=1e-12)
    assert stats["steps"] == 3 * 5 and stats["n"] == 37      # 4 full batches and one batch of 5 per epoch
    assert torch.equal(w0, init_vector(0, D64))               # the global weights are not modified in place


def test_client_update_edge_cases():
    fed = make_fed([1, 3])
    w0 = init_vector(0, D64)
    x1, y1 = fed.client_train(0)
    w, s = client_update(w0, x1, y1, epochs=2, batch_size=10, lr=0.1, seed=0)   # one sample, batch larger than n
    assert s["steps"] == 2 and s["finite"] and not torch.equal(w, w0)
    x3, y3 = fed.client_train(1)
    a, _ = client_update(w0, x3, y3, 1, 0, 0.1, seed=0)        # B = 0 means full batch
    b, _ = client_update(w0, x3, y3, 1, 99, 0.1, seed=1)       # B > n means full batch; the seed is then unused
    assert torch.equal(a, b)
    w, s = client_update(w0, x3, y3, epochs=0, batch_size=2, lr=0.1, seed=0)
    assert torch.equal(w, w0) and s["steps"] == 0
    w, s = client_update(w0, x3[:0], y3[:0], epochs=2, batch_size=2, lr=0.1, seed=0)  # client without data
    assert torch.equal(w, w0) and s["n"] == 0
    w, s = client_update(w0, x3, y3, epochs=1, batch_size=0, lr=float("inf"), seed=0)
    assert not s["finite"]


def test_a_client_without_images_changes_nothing():
    # loading: an empty client becomes an empty array, and the 80/20 split of 0 or 1 image is defined
    x, y = data._to_arrays(dict(images=[], labels=[]))
    assert x.shape == (0, 784) and y.shape == (0,)
    g = np.random.default_rng(0)
    clients = [(x, y), (g.random((10, 784), dtype=np.float32), g.integers(0, 62, 10)),
               (g.random((1, 784), dtype=np.float32), g.integers(0, 62, 1))]
    test = (g.random((3, 784), dtype=np.float32), g.integers(0, 62, 3))
    f = data.build_federation(raw=(clients, test), noise_b=0.5)
    assert list(f.n_tr) == [0, 8, 1] and list(f.n_va) == [0, 2, 0]
    # training: with full-batch steps and all clients selected, an empty client must not change any weight
    cfg = Config(rounds=3, clients_per_round=3, epochs=2, batch_size=0, lr=0.1, seed=0)
    a, ha = run_fedavg(make_fed([0, 5, 7]), cfg)
    b, hb = run_fedavg(make_fed([5, 7]), Config(**{**cfg.__dict__, "clients_per_round": 2}))
    assert torch.equal(a, b) and list(ha["n_returned"]) == [0, 2, 2, 2]
    assert np.isfinite(ha["train_acc"]).all() and np.isnan(ha["client_train_acc"][:, 0]).all()
    assert np.isfinite(ha["client_train_acc"][:, 1:]).all()
    w, h = run_fedavg(f, Config(rounds=2, clients_per_round=3, epochs=1, batch_size=4, lr=0.1, seed=0))
    assert bool(torch.isfinite(w).all()) and list(h["n_returned"]) == [0, 2, 2]


def centralized_gd(fed, w, lr, steps):
    w = w.clone().requires_grad_(True)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(forward(w, fed.x_tr), fed.y_tr)
        (g,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            w -= lr * g
    return w.detach()


def test_fedsgd_equals_centralized_gradient_descent():
    """All clients, E = 1, full batch: the n_k-weighted average of one local gradient step is one
    gradient step on the pooled loss. Unequal client sizes make the weighting matter."""
    fed = make_fed([5, 40, 12, 3, 25])
    cfg = Config(rounds=6, clients_per_round=5, epochs=1, batch_size=0, lr=0.2, seed=1)
    w, _ = run_fedavg(fed, cfg)
    ref = centralized_gd(fed, init_vector(1, D64), 0.2, 6)
    assert (w - ref).abs().max() < 1e-12
    w_uniform, _ = run_fedavg(fed, Config(**{**cfg.__dict__, "weighting": "uniform"}))
    assert (w_uniform - ref).abs().max() > 1e-5   # the check can fail: 1/m weights give a different model


def test_single_client_fedavg_is_local_sgd():
    fed = make_fed([30])
    cfg = Config(rounds=4, clients_per_round=1, epochs=2, batch_size=7, lr=0.1, seed=2)
    w, _ = run_fedavg(fed, cfg)
    ref = init_vector(2, D64)
    x, y = fed.client_train(0)
    for rnd in range(1, 5):
        ref, _ = client_update(ref, x, y, 2, 7, 0.1, core.update_seed(2, rnd, 0))
    assert torch.equal(w, ref)


def test_split_properties():
    for n in [2, 3, 5, 118, 393]:
        tr, va = data.split_indices(n, client_id=4, seed=0)
        assert len(set(tr) & set(va)) == 0 and sorted(list(tr) + list(va)) == list(range(n))
        assert len(va) == min(max(round(0.2 * n), 1), n - 1) and len(tr) >= 1
    a = data.split_indices(100, 4, 0)
    assert all(np.array_equal(p, q) for p, q in zip(a, data.split_indices(100, 4, 0)))
    assert not np.array_equal(a[1], data.split_indices(100, 5, 0)[1])
    tr, va = data.split_indices(1, 0, 0)
    assert len(tr) == 1 and len(va) == 0


def test_client_sampling():
    assert [core.clients_from_fraction(c, 100) for c in [0.0, 0.001, 0.01, 0.025, 0.1, 1.0, 3.0]] == [1, 1, 1, 2, 10, 100, 100]
    fed = make_fed([6] * 20)
    _, h = run_fedavg(fed, Config(rounds=50, clients_per_round=3, epochs=1, batch_size=0, lr=0.01, seed=0))
    sel = h["selected"]
    assert not sel[0].any() and (sel[1:].sum(1) == 3).all()      # exactly m distinct clients per round
    assert sel[1:].sum(0).min() >= 1                             # 150 draws over 20 clients reach every client
    with pytest.raises(ValueError):
        run_fedavg(fed, Config(rounds=1, clients_per_round=21))
    with pytest.raises(ValueError):
        run_fedavg(fed, Config(rounds=1, clients_per_round=0))


def test_runs_are_reproducible():
    fed = make_fed([10, 14, 9, 20])
    cfg = Config(rounds=5, clients_per_round=2, epochs=2, batch_size=4, lr=0.1, seed=3)
    a, ha = run_fedavg(fed, cfg)
    b, hb = run_fedavg(fed, cfg)
    c, _ = run_fedavg(fed, Config(**{**cfg.__dict__, "seed": 4}))
    assert torch.equal(a, b) and np.array_equal(ha["val_loss"], hb["val_loss"]) and not torch.equal(a, c)


class DroppingRunner(core.SequentialRunner):
    """Returns no result for the listed (round, client) pairs."""

    def __init__(self, fed, cfg, drop):
        super().__init__(fed, cfg)
        self.drop = drop

    def __call__(self, rnd, selected, w, lr):
        out = super().__call__(rnd, selected, w, lr)
        return [None if (rnd, r["cid"]) in self.drop else r for r in out]


def test_dropped_clients_are_left_out_and_empty_round_keeps_the_model():
    fed = make_fed([10, 14, 9])
    cfg = Config(rounds=2, clients_per_round=3, epochs=1, batch_size=0, lr=0.1, seed=0)
    drop = {(1, 0), (2, 0), (2, 1), (2, 2)}
    w, h = run_fedavg(fed, cfg, runner=DroppingRunner(fed, cfg, drop))
    w0 = init_vector(0, D64)
    ups = [client_update(w0, *fed.client_train(k), 1, 0, 0.1, 0)[0] for k in (1, 2)]
    expected = aggregate(ups, [14, 9])                    # round 1: clients 1 and 2 only, weights renormalised
    assert torch.allclose(w, expected, atol=1e-14)        # round 2: nobody returned, the model is unchanged
    assert list(h["n_returned"]) == [0, 2, 0] and h["update_norm"][1] > 0 and np.isnan(h["update_norm"][2])


def test_reported_metrics_equal_a_direct_recomputation():
    fed = make_fed([10, 14, 9])
    cfg = Config(rounds=3, clients_per_round=2, epochs=1, batch_size=5, lr=0.1, seed=0)
    w, h = run_fedavg(fed, cfg)
    logits = forward(w, fed.x_tr)
    assert abs(h["train_loss"][-1] - float(torch.nn.functional.cross_entropy(logits, fed.y_tr))) < 1e-12
    assert abs(h["train_acc"][-1] - float((logits.argmax(1) == fed.y_tr).double().mean())) < 1e-12
    per_client = h["client_train_acc"][-1]
    assert abs(np.dot(per_client, fed.n_tr) / fed.n_tr.sum() - h["train_acc"][-1]) < 1e-12


# ---------------------------------------------------------------- real data (skipped when it is absent)
HAVE_DATA = os.path.exists(os.path.join(data.default_data_dir(), "train_data.npy"))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="put train_data.npy and test_data.npy in data/")


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@needs_data
def test_real_data_shapes_and_split(raw):
    clients, (xt, yt) = raw
    assert len(clients) == 100 and sum(len(y) for _, y in clients) == 31825 and len(yt) == 3621
    fed = data.build_federation(raw=raw)
    assert fed.n_clients == 100 and fed.x_tr.shape[1] == 784
    assert (fed.n_tr + fed.n_va == np.array([len(y) for _, y in clients])).all()
    assert abs(fed.n_va.sum() / 31825 - 0.2) < 0.002 and fed.n_va.min() >= 1
    assert float(fed.x_tr.min()) >= 0 and float(fed.x_tr.max()) <= 1
    assert float(fed.x_tr.mean()) < 0.1          # 'invert': the page background is 0


@needs_data
def test_laplace_noise_properties(raw):
    clean = data.build_federation(raw=raw)
    zero = data.build_federation(raw=raw, noise_b=0.0)
    assert torch.equal(clean.x_tr, zero.x_tr) and zero.x_tr_clean is None
    b = 0.5
    noisy = data.build_federation(raw=raw, noise_b=b, noise_seed=0)
    eps = (noisy.x_tr - clean.x_tr).double()
    assert torch.equal(noisy.x_tr_clean, clean.x_tr)
    assert torch.equal(noisy.x_te, clean.x_te)                       # the test data is never perturbed
    assert torch.equal(noisy.y_tr, clean.y_tr)
    assert abs(float(eps.mean())) < 1e-3
    assert abs(float(eps.var()) / (2 * b * b) - 1) < 0.01            # Var = 2 b^2
    assert abs(float(eps.abs().mean()) / b - 1) < 0.01               # E|eps| = b
    assert abs(float((eps.abs() > b).double().mean()) - np.exp(-1)) < 0.002   # P(|eps| > b) = 1/e
    again = data.build_federation(raw=raw, noise_b=b, noise_seed=0)
    other = data.build_federation(raw=raw, noise_b=b, noise_seed=1)
    assert torch.equal(again.x_tr, noisy.x_tr) and not torch.equal(other.x_tr, noisy.x_tr)
    val_clean = data.build_federation(raw=raw, noise_b=b, noise_val=False)
    assert torch.equal(val_clean.x_va, clean.x_va) and torch.equal(val_clean.x_tr, noisy.x_tr)
    clipped = data.build_federation(raw=raw, noise_b=b, clip=True)
    assert float(clipped.x_tr.min()) >= 0 and float(clipped.x_tr.max()) <= 1
    with pytest.raises(ValueError):
        data.build_federation(raw=raw, noise_b=-1.0)


@needs_data
def test_iid_partition_keeps_sizes_and_images(raw):
    writer = data.build_federation(raw=raw)
    iid = data.build_federation(raw=raw, partition="iid")
    assert (writer.n_tr + writer.n_va == iid.n_tr + iid.n_va).all()
    total = lambda f: float(f.x_tr.double().sum() + f.x_va.double().sum())
    assert abs(total(writer) - total(iid)) < 1e-3
    assert torch.equal(torch.bincount(torch.cat([writer.y_tr, writer.y_va]), minlength=62),
                       torch.bincount(torch.cat([iid.y_tr, iid.y_va]), minlength=62))


@needs_data
def test_ray_actors_reproduce_the_sequential_run_and_survive_a_crash(raw):
    from fedavg.ray_clients import RayRunner
    fed = data.build_federation(raw=raw)
    cfg = Config(rounds=3, clients_per_round=4, epochs=1, batch_size=20, lr=0.1, seed=0)
    w_seq, h_seq = run_fedavg(fed, cfg)
    runner = RayRunner(fed, cfg, use_gpu=False)                         # CPU actors, as the sequential loop
    try:
        w_ray, h_ray = run_fedavg(fed, cfg, runner=runner)
        assert torch.equal(w_seq, w_ray)                                # bit-identical weights
        assert len(set(h_ray["timings"][:, 2])) == 12                   # 12 updates ran in 12 different processes
        assert os.getpid() not in set(h_ray["timings"][:, 2].astype(int))
        victim = int(np.flatnonzero(h_seq["selected"][2])[0])
        runner.faults = {(2, victim): "crash"}
        w_crash, h_crash = run_fedavg(fed, cfg, runner=runner)
        assert list(h_crash["n_returned"]) == [0, 4, 3, 4] and bool(torch.isfinite(w_crash).all())
        assert not torch.equal(w_crash, w_seq)
    finally:
        runner.close()


def test_gpu_share_lets_all_actors_of_a_round_run_at_once():
    from fedavg.ray_clients import gpu_share
    assert gpu_share(0, 4) == 0.0 and gpu_share(4, 4) == 1.0 and gpu_share(5, 4) == 1.0
    assert gpu_share(1, 4) == 0.25 and gpu_share(2, 4) == 0.5
    assert gpu_share(3, 4) == 0.5 and gpu_share(2, 3) == 0.5            # G / m (0.75, 0.67) would fit one actor per GPU
    for G in range(1, 9):
        for m in range(1, 21):
            s = gpu_share(G, m)
            assert 0 < s <= 1 and G * int(1.0 / s + 1e-9) >= m          # every GPU hosts floor(1 / s) actors


@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_actors_agree_with_the_cpu_loop_within_rounding(raw):
    from fedavg.ray_clients import RayRunner
    fed = data.build_federation(raw=raw)
    cfg = Config(rounds=3, clients_per_round=4, epochs=1, batch_size=20, lr=0.1, seed=0)
    w_seq, _ = run_fedavg(fed, cfg)
    runner = RayRunner(fed, cfg)
    try:
        w_gpu, _ = run_fedavg(fed, cfg, runner=runner)
        assert runner.gpu_per_actor > 0 and {u["device"] for u in runner.log} == {"cuda"}
        assert torch.allclose(w_seq, w_gpu, atol=1e-5)                  # float32 rounding differs between devices
    finally:
        runner.close()
