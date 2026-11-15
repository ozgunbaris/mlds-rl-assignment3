"""GPU checks of the Ray actors (tests G1 to G6 of the report).

    python -m fedavg.gpu_checks --out results/gpu                # on a machine with CUDA devices (scripts/nomad)
    python -m fedavg.gpu_checks --out results/gpu --only share   # scheduling test with logical GPUs, any machine
    python -m fedavg.gpu_checks --out results/gpu --report       # tables and figure from results/gpu/gpu_checks.json

The measuring part writes <out>/gpu_checks.json after every section, so a partial result survives a failure.
Without CUDA devices the Ray sections still run with --gpus logical GPUs; the actors then train on the CPU.
"""
import argparse
import json
import os
import platform
import time
import traceback
from collections import Counter

import numpy as np
import ray
import torch

from .core import Config, client_update, evaluate, forward, init_vector, run_fedavg, save_run, update_seed
from .data import build_federation
from .ray_clients import RayRunner, gpu_share

OBJECT_STORE = 1_000_000_000  # bytes; the default (30 % of the memory) is not needed for 100 small arrays
BEST = dict(epochs=5, batch_size=20, lr=0.5)


def max_overlap(starts, ends):
    """Largest number of intervals [start, end] that contain a common instant."""
    events = sorted([(s, 1) for s in starts] + [(e, -1) for e in ends], key=lambda t: (t[0], t[1]))
    best = cur = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return int(best)


def run_ray(f, cfg, on_round=None, **kw):
    kw.setdefault("timeout", 600.0)
    runner = RayRunner(f, cfg, object_store_memory=OBJECT_STORE, **kw)
    info = dict(gpu_per_actor=runner.gpu_per_actor, ray_gpus=runner.num_gpus, ray_cpus=runner.num_cpus)
    try:
        t0 = time.time()
        w, h = run_fedavg(f, cfg, runner=runner, on_round=on_round)
        info["seconds"] = time.time() - t0
        log = list(runner.log)
    finally:
        runner.close()
    return w, h, log, info


def summarise(log, h, cfg, info):
    """Numbers of one Ray run, from the per-update log of the runner."""
    m, R = cfg.clients_per_round, cfg.rounds
    by_round = {r: [u for u in log if u["round"] == r] for r in range(1, R + 1)}
    t0 = np.array([u["t_init0"] for u in log]); t1 = np.array([u["t_end"] for u in log])
    alive = [max_overlap([u["t_init0"] for u in us], [u["t_end"] for u in us]) for us in by_round.values() if us]
    training = [max_overlap([u["t_start"] for u in us], [u["t_end"] for u in us]) for us in by_round.values() if us]
    gpus_in_round = [len({g for u in us for g in u["gpu_ids"]}) for us in by_round.values() if us]
    sec = np.asarray(h["seconds"], dtype=float)
    out = dict(info, clients_per_round=m, rounds=R, updates=len(log), distinct_pids=len({u["pid"] for u in log}),
               min_returned=int(np.min(h["n_returned"][1:])),
               devices=dict(Counter(u["device"] for u in log)),
               updates_per_gpu=dict(Counter(",".join(u["gpu_ids"]) or "none" for u in log)),
               gpu_names=dict(Counter(u.get("gpu_name", "none") for u in log)),
               max_alive_at_once=int(max(alive)), rounds_all_alive=int(np.sum(np.array(alive) == m)),
               max_training_at_once=int(max(training)), rounds_all_training=int(np.sum(np.array(training) == m)),
               min_gpus_in_round=int(min(gpus_in_round)), max_gpus_in_round=int(max(gpus_in_round)),
               construct_seconds=float(np.mean([u["t_created"] - u["t_init0"] for u in log])),
               train_seconds=float(np.mean([u["t_end"] - u["t_start"] for u in log])),
               lifetime_seconds=float(np.mean(t1 - t0)),
               round_seconds_first=float(sec[1]), round_seconds=float(sec[2:].mean()) if R > 1 else float(sec[1]))
    if any("gpu_used_bytes" in u for u in log):
        out["gpu_tensor_mb_max"] = max(u.get("gpu_tensor_bytes", 0) for u in log) / 2**20
        out["gpu_used_mb_max"] = max(u.get("gpu_used_bytes", 0) for u in log) / 2**20
        out["gpu_used_mb_min"] = min(u.get("gpu_used_bytes", 0) for u in log if "gpu_used_bytes" in u) / 2**20
    return out


# ------------------------------------------------------------------ sections
def environment(f, a, N):
    """Software, devices and what Ray detects without being told."""
    env = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
               ray=ray.__version__, cuda_available=bool(torch.cuda.is_available()),
               cuda_build=torch.version.cuda, cuda_devices=int(torch.cuda.device_count()),
               cpus_usable=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
               cpus_host=os.cpu_count(), gpus_argument=a.gpus,
               CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
               NVIDIA_VISIBLE_DEVICES=os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    env["device_names"] = [torch.cuda.get_device_name(i) for i in range(env["cuda_devices"])]
    env["device_total_mb"] = [torch.cuda.mem_get_info(i)[1] / 2**20 for i in range(env["cuda_devices"])]
    env["device_used_mb_before"] = [(torch.cuda.mem_get_info(i)[1] - torch.cuda.mem_get_info(i)[0]) / 2**20
                                    for i in range(env["cuda_devices"])]
    try:
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, logging_level="ERROR",
                 object_store_memory=OBJECT_STORE)
        env["ray_detected_gpus"] = float(ray.cluster_resources().get("GPU", 0))
        env["ray_detected_cpus"] = float(ray.cluster_resources().get("CPU", 0))
    finally:
        ray.shutdown()
    N["environment"] = env


@ray.remote
class Probe:
    """An actor that only waits; used to see how many actors Ray runs at once."""

    def __init__(self):
        self.t0 = time.time()

    def hold(self, seconds):
        time.sleep(seconds)
        return self.t0, time.time(), [str(g) for g in ray.get_gpu_ids()]


def share(f, a, N):
    """G1. The GPU share per actor: logical GPUs, probe actors that sleep (no CUDA call)."""
    out = []
    for m, G in [(4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (3, 2), (10, 4)]:
        for rule, frac in [("ceil", gpu_share(G, m)), ("G/m", min(1.0, int(1e4 * G / m) / 1e4))]:
            ray.init(num_cpus=m, num_gpus=G, include_dashboard=False, log_to_driver=False, logging_level="ERROR",
                     object_store_memory=OBJECT_STORE)
            try:
                warm = [Probe.options(num_cpus=1).remote() for _ in range(m)]  # start the worker processes first
                ray.get([p.hold.remote(0.0) for p in warm])
                for p in warm:
                    ray.kill(p)
                time.sleep(0.5)
                t0 = time.time()
                actors = [Probe.options(num_cpus=1, num_gpus=frac).remote() for _ in range(m)]
                pending = {p.hold.remote(1.5): p for p in actors}
                res = []
                while pending:  # as in RayRunner: an actor is killed when its result arrives, which frees its share
                    done, _ = ray.wait(list(pending), num_returns=1)
                    res.append(ray.get(done[0]))
                    ray.kill(pending.pop(done[0]))
                secs = time.time() - t0
            finally:
                ray.shutdown()
            per_gpu = Counter(",".join(r[2]) for r in res)
            out.append(dict(m=m, G=G, rule=rule, share=frac, max_alive=max_overlap([r[0] for r in res], [r[1] for r in res]),
                            seconds=secs, most_actors_on_one_gpu=max(per_gpu.values()), gpus_used=len(per_gpu)))
            print("share", out[-1], flush=True)
    N["share"] = out


def device_path(f, a, N):
    """G2. The actors train on the GPUs Ray assigned; shares of 0.5, 0.25 and 1 GPU; CPU actors for reference."""
    G = a.gpus
    cases = [("gpu_m4", 4, dict(num_gpus=G)), ("gpu_m4_one_device", 4, dict(num_gpus=1)),
             ("gpu_m2", 2, dict(num_gpus=G)), ("gpu_m3_ceil", 3, dict(num_gpus=G)),
             ("gpu_m3_naive", 3, dict(num_gpus=G, gpu_per_actor=min(1.0, int(1e4 * G / 3) / 1e4))),
             ("cpu_m4", 4, dict(num_gpus=G, use_gpu=False))]
    out = {}
    for name, m, kw in cases:
        cfg = Config(rounds=a.rounds_short, clients_per_round=m, seed=0, **BEST)
        w, h, log, info = run_ray(f, cfg, num_cpus=4, **kw)
        out[name] = summarise(log, h, cfg, info)
        out[name]["val_acc"] = float(h["val_acc"][-1])
        print("device_path", name, json.dumps(out[name], default=float), flush=True)
        N["device_path"] = out


def arithmetic(f, a, N):
    """G3. One client update on the CPU and on every CUDA device, from the same weights and the same seed."""
    if not torch.cuda.is_available():
        N["arithmetic"] = dict(skipped="no CUDA device")
        return
    w0 = init_vector(0)
    out = {}
    for k in [0, int(np.argmax(f.n_tr))]:
        x, y = f.client_train(k)
        seed = update_seed(0, 1, k)
        row = dict(client=k, n=int(len(y)))
        for label, E, B in [("one_full_batch_step", 1, 0), ("E5_B20", 5, 20)]:
            ref, st = client_update(w0, x, y, E, B, 0.5, seed)
            r = dict(steps=st["steps"], move_norm=float((ref - w0).norm()), devices={})
            vecs = []
            for d in range(torch.cuda.device_count()):
                dev = f"cuda:{d}"
                v1, _ = client_update(w0.to(dev), x.to(dev), y.to(dev), E, B, 0.5, seed)
                v2, _ = client_update(w0.to(dev), x.to(dev), y.to(dev), E, B, 0.5, seed)
                vecs.append(v1.cpu())
                r["devices"][dev] = dict(name=torch.cuda.get_device_name(d),
                                         max_abs_diff_to_cpu=float((v1.cpu() - ref).abs().max()),
                                         rel_diff_to_cpu=float((v1.cpu() - ref).norm() / (ref - w0).norm()),
                                         repeat_identical=bool(torch.equal(v1, v2)))
            if len(vecs) > 1:
                r["cuda0_vs_cuda1_identical"] = bool(torch.equal(vecs[0], vecs[1]))
                r["cuda0_vs_cuda1_max_abs_diff"] = float((vecs[0] - vecs[1]).abs().max())
            row[label] = r
        out[str(k)] = row
    # forward pass only
    x = f.x_te[:1000]
    lc = forward(w0, x)
    lg = forward(w0.cuda(), x.cuda()).cpu()
    out["forward_logits_max_abs_diff"] = float((lc - lg).abs().max())
    N["arithmetic"] = out
    print("arithmetic", json.dumps(out, default=float), flush=True)


def full_run(f, a, N):
    """G4. A full run with GPU actors against the sequential CPU loop on the same machine, round by round."""
    out = {}
    est = N.get("device_path", {}).get("gpu_m4", {}).get("round_seconds", 0.0)
    R = a.rounds_full if est <= a.max_round_seconds else max(50, a.rounds_full // 3)
    out["rounds"] = R
    out["rounds_reduced_because_round_seconds"] = est if R != a.rounds_full else None
    cfg = Config(rounds=R, clients_per_round=4, seed=0, **BEST)
    ws = []
    t0 = time.time()
    w_seq, h_seq = run_fedavg(f, cfg, on_round=lambda r, wo, res, wn: ws.append(wn.clone()))
    out["seq_seconds"] = time.time() - t0
    save_run(os.path.join(a.out, "seq_cpu_seed0"), cfg, f, w_seq, h_seq, extra=dict(wall_seconds=out["seq_seconds"], ray=False))
    N["full_run"] = out

    # CPU actors on this machine: bit-identical to the sequential loop (test R1 repeated on Linux)
    cfg30 = Config(rounds=a.rounds_short, clients_per_round=4, seed=0, **BEST)
    w_c, h_c, log_c, info_c = run_ray(f, cfg30, num_cpus=4, num_gpus=a.gpus, use_gpu=False)
    out["cpu_actors_identical_to_sequential"] = bool(torch.equal(w_c, ws[a.rounds_short - 1]))
    out["cpu_actors_max_abs_diff"] = float((w_c - ws[a.rounds_short - 1]).abs().max())

    # GPU actors twice: is a GPU run reproducible, and how far is it from the CPU run
    rep = []
    for i in range(2):
        w_g, h_g, log_g, info_g = run_ray(f, cfg30, num_cpus=4, num_gpus=a.gpus)
        rep.append((w_g, {(u["round"], u["cid"]): ",".join(u["gpu_ids"]) for u in log_g}, summarise(log_g, h_g, cfg30, info_g)))
    out["gpu_repeat_identical"] = bool(torch.equal(rep[0][0], rep[1][0]))
    out["gpu_repeat_max_abs_diff"] = float((rep[0][0] - rep[1][0]).abs().max())
    same = [rep[0][1][k] == rep[1][1].get(k) for k in rep[0][1]]
    out["gpu_repeat_same_device_fraction"] = float(np.mean(same))
    out["gpu_short_vs_cpu_max_abs_diff"] = float((rep[0][0] - ws[a.rounds_short - 1]).abs().max())
    out["gpu_repeat_devices"] = rep[0][2]["devices"]
    N["full_run"] = out
    save_json(a, N)

    diffs, rel = [], []

    def hook(r, wo, res, wn):
        diffs.append(float((wn - ws[r - 1]).abs().max()))
        rel.append(float((wn - ws[r - 1]).norm() / ws[r - 1].norm()))
        if r % 25 == 0:
            print(f"full_run round {r} max abs diff {diffs[-1]:.3e}", flush=True)

    w_gpu, h_gpu, log, info = run_ray(f, cfg, on_round=hook, num_cpus=4, num_gpus=a.gpus)
    save_run(os.path.join(a.out, "ray_gpu_seed0"), cfg, f, w_gpu, h_gpu, extra=dict(wall_seconds=info["seconds"], ray=True, gpu=True))
    np.savez(os.path.join(a.out, "gpu_vs_cpu.npz"), max_abs_diff=np.array(diffs), rel_diff=np.array(rel),
             val_acc_cpu=h_seq["val_acc"], val_acc_gpu=h_gpu["val_acc"], train_acc_cpu=h_seq["train_acc"],
             train_acc_gpu=h_gpu["train_acc"], val_loss_cpu=h_seq["val_loss"], val_loss_gpu=h_gpu["val_loss"])
    with open(os.path.join(a.out, "ray_gpu_seed0", "updates.json"), "w") as fh:
        json.dump(log, fh, default=float)
    te_c, te_g = evaluate(w_seq, f.x_te, f.y_te), evaluate(w_gpu, f.x_te, f.y_te)
    pc, pg = forward(w_seq, f.x_te).argmax(1), forward(w_gpu, f.x_te).argmax(1)
    dv = np.abs(h_seq["val_acc"] - h_gpu["val_acc"])
    out.update(summary=summarise(log, h_gpu, cfg, info), test_acc_cpu=te_c["acc"], test_acc_gpu=te_g["acc"],
               val_acc_cpu=float(h_seq["val_acc"][-1]), val_acc_gpu=float(h_gpu["val_acc"][-1]),
               train_acc_cpu=float(h_seq["train_acc"][-1]), train_acc_gpu=float(h_gpu["train_acc"][-1]),
               max_abs_weight_diff={str(r): diffs[r - 1] for r in [1, 2, 5, 10, 30, 100, 200, 300] if r <= R},
               rel_weight_diff={str(r): rel[r - 1] for r in [1, 2, 5, 10, 30, 100, 200, 300] if r <= R},
               first_round_with_difference=int(np.argmax(np.array(diffs) > 0) + 1) if max(diffs) > 0 else None,
               max_val_acc_diff=float(dv.max()), mean_val_acc_diff=float(dv.mean()),
               rounds_with_identical_val_acc=int(np.sum(dv == 0)) - 1,
               last_round_with_identical_val_acc=int(np.max(np.flatnonzero(dv == 0))),
               test_predictions_that_differ=int((pc != pg).sum()), test_n=int(len(pc)),
               weight_norm=float(w_seq.norm()))
    N["full_run"] = out
    print("full_run", json.dumps({k: v for k, v in out.items() if k != "summary"}, default=float), flush=True)


def timing(f, a, N):
    """G5. Seconds per round: sequential CPU loop, CPU actors, GPU actors; and the same update on a warm GPU."""
    out = {}
    for E, R in [(5, 7), (100, 5)]:
        cfg = Config(rounds=R, clients_per_round=4, epochs=E, batch_size=20, lr=0.05, seed=0)
        _, h = run_fedavg(f, cfg)
        row = dict(sequential=float(h["seconds"][2:].mean()),
                   sequential_train=float((h["timings"][:, 4] - h["timings"][:, 3]).mean()))
        for name, kw in [("cpu_actors", dict(use_gpu=False)), ("gpu_actors", {}), ("gpu_actors_one_device", dict(num_gpus=1))]:
            kw = dict(dict(num_gpus=a.gpus), **kw)
            w, hh, log, info = run_ray(f, cfg, num_cpus=4, **kw)
            row[name] = summarise(log, hh, cfg, info)
        out[f"E{E}"] = row
        N["timing"] = out
        print("timing", E, json.dumps(row, default=float), flush=True)
    if torch.cuda.is_available():
        k = int(np.argmax(f.n_tr))
        x, y = f.client_train(k)
        w0 = init_vector(0)
        warm = dict(client=k, n=int(len(y)))
        for label, E, B in [("E5_B20", 5, 20), ("E5_full_batch", 5, 0), ("E5_B200", 5, 200)]:
            r = {}
            for dev in ["cpu", "cuda:0"]:
                xd, yd, wd = x.to(dev), y.to(dev), w0.to(dev)
                client_update(wd, xd, yd, E, B, 0.05, 1)  # warm-up
                ts = []
                for i in range(5):
                    t0 = time.time()
                    _, st = client_update(wd, xd, yd, E, B, 0.05, 1)
                    if dev != "cpu":
                        torch.cuda.synchronize()
                    ts.append(time.time() - t0)
                r[dev] = float(np.mean(ts)); r["steps"] = st["steps"]
            warm[label] = r
        out["warm_update"] = warm
        print("timing warm", json.dumps(warm, default=float), flush=True)
    N["timing"] = out


def faults(f, a, N):
    """G6. A GPU actor process that exits during training: the round continues and the device stays usable."""
    from tests.test_fedavg import DroppingRunner
    cfg = Config(rounds=6, clients_per_round=4, seed=0, **BEST)
    _, h0 = run_fedavg(f, cfg)
    sel = {r: [int(k) for k in np.flatnonzero(h0["selected"][r])] for r in range(1, 7)}
    fl = {(2, sel[2][0]): "crash", (4, sel[4][1]): "nan"}
    fl.update({(5, k): "crash" for k in sel[5]})
    w, h, log, info = run_ray(f, cfg, num_cpus=4, num_gpus=a.gpus, faults=fl)
    w_ref, _ = run_fedavg(f, cfg, runner=DroppingRunner(f, cfg, set(fl)))
    used = {r: [u.get("gpu_used_bytes", 0) / 2**20 for u in log if u["round"] == r] for r in range(1, 7)}
    N["faults"] = dict(returned=[int(v) for v in h["n_returned"]], devices=dict(Counter(u["device"] for u in log)),
                       devices_round6=dict(Counter(u["device"] for u in log if u["round"] == 6)),
                       max_abs_diff_to_cpu_reference=float((w - w_ref).abs().max()),
                       finite=bool(torch.isfinite(w).all()), round_seconds=[float(v) for v in h["seconds"]],
                       gpu_used_mb_max_by_round={str(r): (max(v) if v else None) for r, v in used.items()})
    print("faults", json.dumps(N["faults"], default=float), flush=True)


def sensitivity(f, a, N):
    """G4 control, CPU only. The sequential loop is repeated from initial weights that differ by one float32
    rounding step; the growth of that difference is compared with the GPU-against-CPU difference of full_run."""
    out, curves = {}, {}
    for lr in [0.5, 0.05]:
        cfg = Config(rounds=a.rounds_full, clients_per_round=4, seed=0, epochs=5, batch_size=20, lr=lr)
        w0 = init_vector(0)
        ws = []
        w_a, h_a = run_fedavg(f, cfg, on_round=lambda r, wo, res, wn: ws.append(wn.clone()))
        up = torch.nextafter(w0, torch.full_like(w0, float("inf")))
        one = w0.clone(); one[-1] = up[-1]  # the bias of the last class
        for label, start in [("all_weights_one_step", up), ("one_weight_one_step", one)]:
            if lr != 0.5 and label != "all_weights_one_step":
                continue
            diffs, rel = [], []

            def hook(r, wo, res, wn):
                diffs.append(float((wn - ws[r - 1]).abs().max()))
                rel.append(float((wn - ws[r - 1]).norm() / ws[r - 1].norm()))

            w_b, h_b = run_fedavg(f, cfg, on_round=hook, init_vec=start)
            pa, pb = forward(w_a, f.x_te).argmax(1), forward(w_b, f.x_te).argmax(1)
            dv = np.abs(h_a["val_acc"] - h_b["val_acc"])
            key = f"lr{lr}_{label}"
            curves[key + "_max_abs"] = np.array(diffs); curves[key + "_rel"] = np.array(rel)
            out[key] = dict(lr=lr, start_max_abs_diff=float((start - w0).abs().max()),
                            start_rel_diff=float((start - w0).norm() / w0.norm()),
                            max_abs_weight_diff={str(r): diffs[r - 1] for r in [1, 2, 5, 10, 30, 100, 200, 300] if r <= cfg.rounds},
                            rel_weight_diff={str(r): rel[r - 1] for r in [1, 2, 5, 10, 30, 100, 200, 300] if r <= cfg.rounds},
                            test_acc_a=evaluate(w_a, f.x_te, f.y_te)["acc"], test_acc_b=evaluate(w_b, f.x_te, f.y_te)["acc"],
                            val_acc_a=float(h_a["val_acc"][-1]), val_acc_b=float(h_b["val_acc"][-1]),
                            max_val_acc_diff=float(dv.max()), mean_val_acc_diff=float(dv.mean()),
                            last_round_with_identical_val_acc=int(np.max(np.flatnonzero(dv == 0))),
                            test_predictions_that_differ=int((pa != pb).sum()), test_n=int(len(pa)))
            print("sensitivity", key, json.dumps(out[key], default=float), flush=True)
    np.savez(os.path.join(a.out, "sensitivity.npz"), **curves)
    N["sensitivity"] = out


SECTIONS = dict(environment=environment, share=share, device_path=device_path, arithmetic=arithmetic,
                full_run=full_run, timing=timing, faults=faults, sensitivity=sensitivity)
NEEDS_REAL_GPU_COUNT = {"share"}  # declares more logical GPUs than exist, which Ray refuses when CUDA devices are visible


def save_json(a, N):
    with open(os.path.join(a.out, "gpu_checks.json"), "w") as fh:
        json.dump(N, fh, indent=1, default=float)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=os.path.join("results", "gpu"))
    p.add_argument("--data", default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--gpus", type=int, default=None, help="GPUs given to Ray (default: CUDA devices of this machine)")
    p.add_argument("--rounds-short", type=int, default=30)
    p.add_argument("--rounds-full", type=int, default=300)
    p.add_argument("--max-round-seconds", type=float, default=8.0)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        from .gpu_report import write_report
        write_report(a.out)
        return
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    if a.gpus is None:
        a.gpus = torch.cuda.device_count()
    path = os.path.join(a.out, "gpu_checks.json")
    N = json.load(open(path)) if os.path.exists(path) else {}
    f = build_federation(a.data)
    default = [s for s in SECTIONS if s not in NEEDS_REAL_GPU_COUNT] if torch.cuda.is_available() else list(SECTIONS)
    for name in (a.only or default):
        t0 = time.time()
        try:
            SECTIONS[name](f, a, N)
            N.setdefault("errors", {}).pop(name, None)
        except Exception:
            N.setdefault("errors", {})[name] = traceback.format_exc()
            print(f"[{name}] FAILED\n{traceback.format_exc()}", flush=True)
            if ray.is_initialized():
                ray.shutdown()
        N.setdefault("section_seconds", {})[name] = time.time() - t0
        save_json(a, N)
        print(f"[{name}] {time.time() - t0:.0f} s", flush=True)
    print("GPU_CHECKS_DONE", flush=True)


if __name__ == "__main__":
    main()
