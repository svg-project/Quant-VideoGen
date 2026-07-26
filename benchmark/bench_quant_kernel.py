#!/usr/bin/env python3
"""
Controlled pure-quantize-kernel benchmark — median per-(K+V)-layer latency only.

WARMUP IS EXCLUDED: `WARMUP` calls run and are discarded (compile/k-means/autotune warm-up),
then `ITERS` calls are timed with a per-call cuda.synchronize() (wall-clock per call), so the
reported number is the steady warm kernel only — no JIT, no first-call autotune, no offload.

Measures compress_kv_cache(K,V) on a fixed single-DiT-layer tensor [1,32,29640,128] (LongCat
480p, 73-frame ctx, 32 heads × 128 head_dim).
Run:  conda activate qvg && python bench_quant_kernel.py
"""
import os, sys, json, time, statistics

import torch

QVG = os.environ.get("QVG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, QVG)

SHAPE = (1, 32, 29640, 128)            # [B, H, S, D] one DiT layer (K, and same for V)
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))   # excluded from timing
ITERS = int(os.environ.get("BENCH_ITERS", "20"))    # timed
DTYPE = torch.bfloat16
DEV = "cuda"

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


# (label, quant_type, QuantizeConfig kwargs, env to set)
ARMS = [
    ("QVG (S1,B64)",      "triton-nstages-kmeans-int2",
        dict(num_prq_stages=1, quant_block_size=64, kmeans_max_iters=5), {}),
    ("QVG-Pro (S4,B16)",  "triton-nstages-kmeans-int2",
        dict(num_prq_stages=4, quant_block_size=16, kmeans_max_iters=5), {}),
]


def bench(label, qtype, cfg_kwargs, env):
    for kk, vv in env.items():
        os.environ[kk] = vv
    cfg = QuantizeConfig(quant_type=qtype, **cfg_kwargs)
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

    med = statistics.median(times_ms)
    print(f"{label:20s} | per-(K+V)-layer median {med:8.2f} ms")
    return dict(arm=label, per_layer_ms_median=round(med, 3))


def main():
    print(f"shape={SHAPE} dtype=bf16 | WARMUP={WARMUP} (EXCLUDED) ITERS={ITERS} (timed) | "
          f"data=N(0,1) seed=0 | gpu={torch.cuda.get_device_name(0)}")
    results = []
    for arm in ARMS:
        results.append(bench(*arm))
        torch.cuda.empty_cache()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_quant_kernel.json")
    json.dump({"shape": SHAPE, "warmup": WARMUP, "iters": ITERS, "results": results},
              open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
