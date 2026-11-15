"""Tables and the figure of the GPU checks, written from the stored numbers.

    python -m fedavg.gpu_checks --report

Reads results/gpu/gpu_checks.json (cluster job, scripts/nomad) and results/gpu_share/gpu_checks.json (local
scheduling test with logical GPUs). Writes results/validation/gpu_checks.md and results/figures/gpu_vs_cpu.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOCAL = os.path.join("results", "gpu_share")
VAL = os.path.join("results", "validation")
FIG = os.path.join("results", "figures")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def sci(v):
    return "0" if v == 0 else f"{v:.1e}"


def load(path):
    p = os.path.join(path, "gpu_checks.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def write_report(out):
    N, L = load(out), load(LOCAL)
    os.makedirs(VAL, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    md = []
    env = N.get("environment", {})
    if env:
        md.append("## GPU checks of the Ray actors (R5 to R10)\n\n"
                  f"Machine: one Nomad allocation on mlds-deepdish5 with {env['cpus_usable']} CPU cores and "
                  f"{env['cuda_devices']} GPUs ({', '.join(sorted(set(env['device_names'])))}, "
                  f"{env['device_total_mb'][0] / 1024:.1f} GiB each). Software: Python {env['python']}, "
                  f"PyTorch {env['torch']}, Ray {env['ray']}. Ray counted {env['ray_detected_gpus']:.0f} GPUs without being told.\n")

    # ---- R5 share rule
    if "share" in L:
        rows = []
        cases = sorted({(r["m"], r["G"]) for r in L["share"]}, key=lambda t: (t[0], t[1]))
        for m, G in cases:
            c = next(r for r in L["share"] if (r["m"], r["G"], r["rule"]) == (m, G, "ceil"))
            n = next(r for r in L["share"] if (r["m"], r["G"], r["rule"]) == (m, G, "G/m"))
            rows.append([m, G, f"{n['share']:.4f}", n["max_alive"], f"{n['seconds']:.1f}", f"{c['share']:.4f}", c["max_alive"], f"{c['seconds']:.1f}"])
        md.append("### R5. GPU share per actor\n\n"
                  "Ray places a fractional GPU request on one device. Prediction: a share of G/m lets all m actors run at "
                  "once only when m/G is an integer, whereas the share 1/ceil(m/G) always does. Probe actors that sleep "
                  "1.5 s were run with logical GPUs on the laptop (no CUDA call is made).\n\n"
                  + table(["actors m", "GPUs G", "share G/m", "actors alive at once", "seconds", "share 1/ceil(m/G)", "actors alive at once", "seconds"], rows))
    dp = N.get("device_path", {})
    if dp:
        c, n = dp["gpu_m3_ceil"], dp["gpu_m3_naive"]
        md.append(f"On the 2 real GPUs with 3 clients per round ({c['rounds']} rounds of training): share {c['gpu_per_actor']} gives "
                  f"{c['max_alive_at_once']} actors alive at once in {c['rounds_all_alive']} of {c['rounds']} rounds and {c['round_seconds']:.2f} s per round; "
                  f"share {n['gpu_per_actor']} gives at most {n['max_alive_at_once']} actors alive at once and {n['round_seconds']:.2f} s per round. "
                  f"Both runs end with the same validation accuracy ({c['val_acc']:.4f} and {n['val_acc']:.4f}).\n")

        # ---- R6 device path
        names = [("gpu_m4", "4 clients, 2 GPUs"), ("gpu_m4_one_device", "4 clients, 1 GPU"), ("gpu_m2", "2 clients, 2 GPUs"),
                 ("gpu_m3_ceil", "3 clients, 2 GPUs"), ("cpu_m4", "4 clients, GPUs not requested")]
        rows = []
        for key, label in names:
            d = dp[key]
            per = ", ".join(f"GPU {g}: {v}" for g, v in sorted(d["updates_per_gpu"].items())) if d["gpu_per_actor"] else "none"
            rows.append([label, d["gpu_per_actor"], f"{d['devices'].get('cuda', 0)} of {d['updates']}", per,
                         f"{d['min_gpus_in_round']}", f"{d['rounds_all_alive']} of {d['rounds']}", d["distinct_pids"],
                         f"{d.get('gpu_tensor_mb_max', 0):.0f}", f"{d.get('gpu_used_mb_max', 0):.0f}"])
        md.append("### R6. The actors train on the GPUs that Ray assigned\n\n"
                  "Prediction: every update runs on a CUDA device, the updates are spread evenly over the devices, all "
                  "actors of a round are alive at once, and every update has its own process.\n\n"
                  + table(["setting", "GPU share per actor", "updates on CUDA", "updates per device", "devices used per round (min)",
                           "rounds with all actors alive at once", "distinct processes", "largest memory allocated by PyTorch in one actor (MiB)",
                           "largest device memory in use (MiB)"], rows)
                  + f"\nThe device memory in use before the job was {env['device_used_mb_before'][0]:.0f} MiB per GPU. "
                  "The device figure includes the CUDA context of every process on that GPU.\n")

    # ---- R7 arithmetic
    ar = N.get("arithmetic", {})
    if ar and "skipped" not in ar:
        rows = []
        for k in [key for key in ar if key.isdigit()]:
            for label, text in [("one_full_batch_step", "1 full-batch step"), ("E5_B20", "E = 5, B = 20")]:
                r = ar[k][label]
                d0 = r["devices"]["cuda:0"]
                rows.append([ar[k]["client"], ar[k]["n"], text, r["steps"], sci(d0["max_abs_diff_to_cpu"]), sci(d0["rel_diff_to_cpu"]),
                             all(d["repeat_identical"] for d in r["devices"].values()), r.get("cuda0_vs_cuda1_identical", "n/a")])
        md.append("### R7. One client update on the CPU and on the GPUs\n\n"
                  "Prediction: the batch order is drawn on the CPU from the update seed, so a GPU update differs from the "
                  "CPU update only by float32 rounding; repeating it on the same device model gives identical bits.\n\n"
                  + table(["client", "training images", "update", "SGD steps", "largest weight difference GPU vs CPU",
                           "difference relative to the distance moved", "repeat on the same GPU identical", "GPU 0 and GPU 1 identical"], rows)
                  + f"\nLargest difference of the logits of 1,000 test images between CPU and GPU: {sci(ar['forward_logits_max_abs_diff'])}.\n")

    # ---- R8 full run
    fr = N.get("full_run", {})
    S = N.get("sensitivity") or L.get("sensitivity", {})
    s_where = "the same cluster machine" if N.get("sensitivity") else "the laptop"
    if "summary" in fr:
        rounds = [r for r in ["1", "2", "5", "10", "30", "100", "200", "300"] if r in fr["max_abs_weight_diff"]]
        rows = [["GPU actors vs CPU loop, learning rate 0.5"] + [sci(fr["max_abs_weight_diff"][r]) for r in rounds]]
        lab = {"lr0.5_all_weights_one_step": "CPU loop twice, all initial weights moved one rounding step, learning rate 0.5",
               "lr0.5_one_weight_one_step": "CPU loop twice, one initial bias moved one rounding step, learning rate 0.5",
               "lr0.05_all_weights_one_step": "CPU loop twice, all initial weights moved one rounding step, learning rate 0.05"}
        for key, text in lab.items():
            if key in S:
                rows.append([text] + [sci(S[key]["max_abs_weight_diff"][r]) for r in rounds])
        md.append("### R8. A full run with GPU actors against the CPU loop\n\n"
                  f"Setting: 4 clients per round, {fr['rounds']} rounds, E = 5, B = 20, learning rate 0.5, seed 0, both on the cluster machine. "
                  "Prediction: the two runs start identical, differ by rounding after round 1, and the difference then grows because "
                  "every later update starts from slightly different weights. The control repeats the CPU loop from initial weights "
                  f"that differ by one float32 rounding step (run on {s_where}).\n\n"
                  + table(["largest absolute weight difference after round"] + rounds, rows))
        rows = [["CPU loop", f"{fr['train_acc_cpu']:.4f}", f"{fr['val_acc_cpu']:.4f}", f"{fr['test_acc_cpu']:.4f}", "", ""],
                ["GPU actors", f"{fr['train_acc_gpu']:.4f}", f"{fr['val_acc_gpu']:.4f}", f"{fr['test_acc_gpu']:.4f}",
                 f"{fr['test_predictions_that_differ']} of {fr['test_n']}", f"{fr['max_val_acc_diff']:.4f} / {fr['mean_val_acc_diff']:.4f}"]]
        for key, text in [("lr0.5_all_weights_one_step", "CPU loop, initial weights one rounding step away")]:
            if key in S:
                s = S[key]
                rows.append([text, "", f"{s['val_acc_b']:.4f}", f"{s['test_acc_b']:.4f}", f"{s['test_predictions_that_differ']} of {s['test_n']}",
                             f"{s['max_val_acc_diff']:.4f} / {s['mean_val_acc_diff']:.4f}"])
        md.append("\n" + table(["run", "final training accuracy", "final validation accuracy", "test accuracy",
                                 "test predictions that differ from the CPU loop", "validation accuracy gap over the rounds (largest / mean)"], rows))
        md.append(f"\nCPU actors on the cluster machine are identical to the CPU loop after {N['device_path']['cpu_m4']['rounds']} rounds: "
                  f"{fr['cpu_actors_identical_to_sequential']} (largest difference {sci(fr['cpu_actors_max_abs_diff'])}). "
                  f"Two GPU runs with the same seed are identical: {fr['gpu_repeat_identical']} (largest difference {sci(fr['gpu_repeat_max_abs_diff'])}); "
                  f"{100 * fr['gpu_repeat_same_device_fraction']:.0f}% of their updates ran on the same device index in both runs. "
                  f"The full GPU run used {fr['summary']['updates']} updates in {fr['summary']['distinct_pids']} processes, all on CUDA: "
                  f"{fr['summary']['devices'].get('cuda', 0) == fr['summary']['updates']}.\n")
        mac = os.path.join("results", "ray", "seed0")
        if os.path.exists(os.path.join(mac, "final_weights.pt")) and os.path.exists(os.path.join(out, "seq_cpu_seed0", "final_weights.pt")):
            import pandas as pd
            import torch
            wa, wb = torch.load(os.path.join(mac, "final_weights.pt")), torch.load(os.path.join(out, "seq_cpu_seed0", "final_weights.pt"))
            ha, hb = pd.read_csv(os.path.join(mac, "history.csv")), pd.read_csv(os.path.join(out, "seq_cpu_seed0", "history.csv"))
            same = np.flatnonzero(ha.val_acc.values == hb.val_acc.values)
            ta = json.load(open(os.path.join(mac, "summary.json")))["test_acc"]
            md.append(f"\nThe same run on two CPUs (laptop, Apple M5, against the cluster machine, x86-64): largest weight difference after 300 rounds "
                      f"{sci(float((wa - wb).abs().max()))}, test accuracy {ta:.4f} against {fr['test_acc_cpu']:.4f}, "
                      f"per-round validation accuracy identical in the first {int(np.argmax(ha.val_acc.values != hb.val_acc.values))} rounds.\n")
        figure(out, fr, S, N.get("sensitivity") is not None)

    # ---- R9 timing
    tm = N.get("timing", {})
    if tm:
        rows = []
        for E in ["E5", "E100"]:
            if E not in tm:
                continue
            t = tm[E]
            rows.append([E[1:], "sequential CPU loop", f"{t['sequential']:.2f}", "", f"{t['sequential_train']:.3f}"])
            for key, text in [("cpu_actors", "Ray, CPU actors"), ("gpu_actors", "Ray, GPU actors on 2 GPUs"), ("gpu_actors_one_device", "Ray, GPU actors on 1 GPU")]:
                d = t[key]
                rows.append([E[1:], text, f"{d['round_seconds']:.2f}", f"{d['construct_seconds']:.3f}", f"{d['train_seconds']:.3f}"])
        md.append("### R9. Wall-clock time with CPU actors and GPU actors\n\n"
                  "Prediction: this network has 108,478 weights and a batch has 20 images, so a GPU step is dominated by the fixed cost of "
                  "launching its operations and a GPU actor is not faster than a CPU actor; each new actor process must also "
                  "initialise CUDA.\n\n"
                  + table(["E", "runner", "seconds per round", "seconds in the actor constructor", "seconds of training per client"], rows))
        if "warm_update" in tm:
            w = tm["warm_update"]
            rows = [[text, w[key]["steps"], f"{w[key]['cpu']:.3f}", f"{w[key]['cuda:0']:.3f}", f"{w[key]['cpu'] / w[key]['cuda:0']:.2f}"]
                    for key, text in [("E5_B20", "B = 20"), ("E5_B200", "B = 200"), ("E5_full_batch", "full batch")]]
            md.append(f"\nThe same update (client {w['client']}, {w['n']} images, E = 5) in one process after a warm-up call:\n\n"
                      + table(["batch size", "SGD steps", "CPU seconds (1 thread)", "GPU seconds", "CPU time / GPU time"], rows))

    # ---- R10 faults
    fa = N.get("faults", {})
    if fa:
        md.append("### R10. Failing GPU actors\n\n"
                  "Round 2: one actor process exits during training. Round 4: one actor returns non-finite weights. Round 5: all four exit. "
                  f"Updates used in rounds 1 to 6: {fa['returned'][1:]}. Devices of the round-6 updates: {fa['devices_round6']}. "
                  f"Largest difference to a CPU reference run that omits the same clients: {sci(fa['max_abs_diff_to_cpu_reference'])}. "
                  f"All weights finite: {fa['finite']}. Largest device memory in use per round (MiB): "
                  + ", ".join(f"{r}: {v:.0f}" if v is not None else f"{r}: no update" for r, v in fa["gpu_used_mb_max_by_round"].items()) + ".\n")
    if N.get("errors"):
        md.append("### Sections that failed\n\n" + "\n".join(f"- {k}" for k in N["errors"]) + "\n")
    with open(os.path.join(VAL, "gpu_checks.md"), "w") as fh:
        fh.write("\n".join(md))
    print("wrote", os.path.join(VAL, "gpu_checks.md"))


def figure(out, fr, S, same_machine):
    z = np.load(os.path.join(out, "gpu_vs_cpu.npz"))
    sp = os.path.join(out if same_machine else LOCAL, "sensitivity.npz")
    s = np.load(sp) if os.path.exists(sp) else {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.7))
    r = np.arange(1, len(z["max_abs_diff"]) + 1)
    ax[0].semilogy(r, z["max_abs_diff"], color="C3", lw=1.6, label="GPU actors vs CPU loop (rate 0.5)")
    for key, color, ls, text in [("lr0.5_all_weights_one_step_max_abs", "C0", "-", "CPU loop twice, start 1 rounding step apart (rate 0.5)"),
                                 ("lr0.05_all_weights_one_step_max_abs", "C2", "--", "the same with learning rate 0.05"),
                                 ("lr0.5_one_weight_one_step_max_abs", "0.4", ":", "CPU loop twice, one bias 1 rounding step apart (rate 0.5)")]:
        if key in s:
            v = np.asarray(s[key], dtype=float)
            ax[0].semilogy(np.arange(1, len(v) + 1), np.where(v > 0, v, np.nan), color=color, ls=ls, lw=1.2, label=text)
    ax[0].set_xlabel("communication round"); ax[0].set_ylabel("largest absolute weight difference")
    ax[0].set_xscale("log"); ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    ax[0].set_title("Difference between two runs with the same seed", fontsize=10)
    rr = np.arange(len(z["val_acc_cpu"]))
    ax[1].plot(rr, 100 * z["val_acc_cpu"], color="C0", lw=1.0, label="CPU loop")
    ax[1].plot(rr, 100 * z["val_acc_gpu"], color="C3", lw=1.0, alpha=0.8, label="GPU actors")
    ax[1].set_ylim(55, 85); ax[1].set_xlabel("communication round"); ax[1].set_ylabel("validation accuracy (%)")
    ax2 = ax[1].twinx()
    ax2.fill_between(rr, 0, 100 * np.abs(z["val_acc_cpu"] - z["val_acc_gpu"]), color="0.6", alpha=0.6, step="mid", label="absolute gap")
    ax2.set_ylim(0, 12); ax2.set_ylabel("gap (points)")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left"); ax[1].grid(alpha=0.3)
    ax[1].set_title("Validation accuracy of the two runs", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpu_vs_cpu.png"), dpi=170)
    plt.close(fig)
    print("wrote", os.path.join(FIG, "gpu_vs_cpu.png"))
