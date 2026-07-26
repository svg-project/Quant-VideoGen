#!/usr/bin/env python3
"""
Sweep kmeans_max_iters and measure median per-(K+V)-layer latency for the two QVG arms.

WARMUP IS EXCLUDED: `WARMUP` calls run and are discarded (compile/k-means/autotune warm-up),
then `ITERS` calls are timed with a per-call cuda.synchronize(), so the reported number is the
steady warm kernel only. Produces a labeled latency-vs-iters plot.

Run:  conda activate qvg && python bench_kmeans_sweep.py
"""
import os, sys, json, time, statistics

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QVG = os.environ.get("QVG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, QVG)

SHAPE = (1, 32, 29640, 128)            # [B, H, S, D] one DiT layer (K, and same for V)
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))   # excluded from timing
ITERS = int(os.environ.get("BENCH_ITERS", "20"))    # timed
DTYPE = torch.bfloat16
DEV = "cuda"

# kmeans_max_iters values to sweep (repo uses 2/4/100 across experiments)
KMEANS_ITERS = [1, 2, 4, 8, 16, 32, 64, 100]

from quant_videogen.compress import compress_kv_cache, get_quantize_fn
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
try:                                       # OSCAR not benchmarked here, but import may set layer state
    import quant_videogen.oscar as _oscar
    _oscar.CURRENT_LAYER = 0
except Exception:
    pass


def make_kv():
    g = torch.Generator(device=DEV).manual_seed(0)
    k = torch.randn(*SHAPE, generator=g, device=DEV, dtype=DTYPE)
    v = torch.randn(*SHAPE, generator=g, device=DEV, dtype=DTYPE)
    return k, v


# (label, quant_type, base QuantizeConfig kwargs)  — kmeans_max_iters injected per sweep point
ARMS = [
    ("QVG (S1,B64)",     "triton-nstages-kmeans-int2", dict(num_prq_stages=1, quant_block_size=64)),
    ("QVG-Pro (S4,B16)", "triton-nstages-kmeans-int2", dict(num_prq_stages=4, quant_block_size=16)),
]


def bench(qtype, cfg_kwargs, kmeans_iters):
    cfg = QuantizeConfig(quant_type=qtype, kmeans_max_iters=kmeans_iters, **cfg_kwargs)
    qfn = get_quantize_fn(qtype, cfg)
    k, v = make_kv()

    for _ in range(WARMUP):                       # ← warm-up, NOT timed
        compress_kv_cache(k, v, qtype, cfg, qfn)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(ITERS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        compress_kv_cache(k, v, qtype, cfg, qfn)
        torch.cuda.synchronize(); times_ms.append((time.perf_counter() - t0) * 1000.0)

    return statistics.median(times_ms)


def main():
    print(f"shape={SHAPE} dtype=bf16 | WARMUP={WARMUP} (EXCLUDED) ITERS={ITERS} (timed) | "
          f"data=N(0,1) seed=0 | gpu={torch.cuda.get_device_name(0)}")
    print(f"sweeping kmeans_max_iters={KMEANS_ITERS}\n")

    results = {}
    for label, qtype, cfg_kwargs in ARMS:
        meds = []
        for it in KMEANS_ITERS:
            med = bench(qtype, cfg_kwargs, it)
            meds.append(med)
            print(f"{label:20s} | kmeans_iters={it:4d} | median {med:8.2f} ms")
            torch.cuda.empty_cache()
        results[label] = meds
        print()

    here = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(here, "bench_kmeans_sweep.json")
    json.dump({"shape": SHAPE, "warmup": WARMUP, "iters": ITERS,
               "kmeans_iters": KMEANS_ITERS, "median_ms": results},
              open(out_json, "w"), indent=2)
    print(f"saved → {out_json}")

    # ---- plot (one panel per arm) ----
    colors = ["tab:blue", "tab:orange"]
    items = list(results.items())
    fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 6))
    if len(items) == 1:
        axes = [axes]
    for ax, (label, meds), color in zip(axes, items, colors):
        ax.plot(KMEANS_ITERS, meds, marker="o", linewidth=2, color=color)
        for x, y in zip(KMEANS_ITERS, meds):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KMEANS_ITERS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("kmeans_max_iters")
        ax.set_ylabel("per-(K+V)-layer median latency (ms)")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.margins(y=0.12)
    fig.suptitle(f"QVG quantize-kernel latency vs kmeans_max_iters — shape={SHAPE}, {torch.cuda.get_device_name(0)}")
    fig.tight_layout()
    out_png = os.path.join(here, "bench_kmeans_sweep.png")
    fig.savefig(out_png, dpi=150)
    print(f"saved → {out_png}")


if __name__ == "__main__":
    main()
