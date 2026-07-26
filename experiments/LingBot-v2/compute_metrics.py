#!/usr/bin/env python
"""Compare quantized runs against the bf16 reference for LingBot-World-v2.

Reads each run dir's:
  - video.mp4       : encoded video output (any *.mp4 in the dir works)
  - peak_mem_mb.txt : end-to-end peak GPU memory (MB)
  - run.log         : contains a "[KV-CACHE] ... Total KV Cache: X MB" line

Metrics are computed on codec-decoded frames, so they include H.264
encoding noise on both sides of the comparison.

Computes PSNR / SSIM / LPIPS of each quantized run vs the bf16 reference
(averaged over frames), and prints a summary table with memory + KV reductions.

Usage: python compute_metrics.py <base_dir>   # base has bf16/ int4/ int2/
"""
import json
import re
import sys
from pathlib import Path

import cv2
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

BASE = Path(sys.argv[1])
RUNS = ["bf16", "int4", "int2"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_video(run: str):
    """Return (N, C, H, W) float in [0,1] decoded from the run's mp4, or None."""
    d = BASE / run
    p = d / "video.mp4"
    if not p.exists():
        mp4s = sorted(d.glob("*.mp4"))
        if not mp4s:
            return None
        p = mp4s[0]
    cap = cv2.VideoCapture(str(p))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(torch.from_numpy(frame[:, :, ::-1].copy()))  # BGR -> RGB
    cap.release()
    if not frames:
        return None
    return torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0


def read_peak_mem(run: str):
    p = BASE / run / "peak_mem_mb.txt"
    return float(p.read_text().strip()) if p.exists() else None


def read_kv_mem(run: str):
    """Parse the LAST '[KV-CACHE]' line: (per_layer, total_kv, gen_peak) MB.

    gen_peak is the generation-phase stable footprint printed right after the
    last quantization (KV + non-KV resident, EXCLUDING the one-shot VAE-decode
    spike) — this is what we report as end-to-end memory.
    """
    p = BASE / run / "run.log"
    if not p.exists():
        return None, None, None
    txt = p.read_text()
    per = re.findall(r"Per Layer:\s*([\d.]+)\s*MB", txt)
    tot = re.findall(r"Total KV Cache:\s*([\d.]+)\s*MB", txt)
    peak = re.findall(r"Peak:\s*([\d.]+)\s*MB", txt)
    return (float(per[-1]) if per else None,
            float(tot[-1]) if tot else None,
            float(peak[-1]) if peak else None)


@torch.no_grad()
def metrics(ref: torch.Tensor, cmp: torch.Tensor):
    """PSNR/SSIM/LPIPS between two (N,C,H,W) [0,1] videos.

    Uses torchmetrics accumulate-then-compute so PSNR is the global value over
    all frames (finite even when the pre-quantization frames are identical to
    the reference — a per-batch average would hit inf on those MSE==0 frames).
    """
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEV)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEV)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(DEV)
    n = min(ref.shape[0], cmp.shape[0])
    for i in range(0, n, 16):                           # batch frames
        r = ref[i:i + 16].to(DEV)
        c = cmp[i:i + 16].to(DEV)
        psnr.update(c, r)
        ssim.update(c, r)
        lpips.update(c, r)
    return psnr.compute().item(), ssim.compute().item(), lpips.compute().item()


def main():
    ref = load_video("bf16")
    assert ref is not None, f"missing bf16 reference at {BASE/'bf16'}"
    _, ref_kv, ref_peak = read_kv_mem("bf16")   # E2E = gen-phase stable Peak
    ref_vae_peak = read_peak_mem("bf16")        # whole-run peak (incl VAE), for reference

    rows = []
    for run in RUNS:
        vid = load_video(run)
        per_kv, tot_kv, peak = read_kv_mem(run)
        vae_peak = read_peak_mem(run)
        if run == "bf16":
            row = dict(run=run, psnr=float("inf"), ssim=1.0, lpips=0.0,
                       peak_mb=peak, vae_peak_mb=vae_peak,
                       per_layer_kv_mb=per_kv, total_kv_mb=tot_kv)
        else:
            if vid is None:
                print(f"[skip] {run}: no mp4 found")
                continue
            p, s, l = metrics(ref, vid)
            row = dict(run=run, psnr=p, ssim=s, lpips=l, peak_mb=peak,
                       vae_peak_mb=vae_peak, per_layer_kv_mb=per_kv, total_kv_mb=tot_kv)
        # reductions vs bf16
        row["e2e_mem_reduction"] = (ref_peak / peak) if (peak and ref_peak) else None
        row["kv_mem_reduction"] = (ref_kv / tot_kv) if (tot_kv and ref_kv) else None
        rows.append(row)

    # ---- print table ----
    print("\n" + "=" * 108)
    print(f"{'Run':6} | {'PSNR(dB)':>9} | {'SSIM':>7} | {'LPIPS':>7} | "
          f"{'KV/layer(MB)':>12} | {'KV total(MB)':>12} | {'KV↓':>6} | "
          f"{'E2E-gen(MB)':>12} | {'E2E↓':>6}")
    print("-" * 108)
    for r in rows:
        psnr_s = "  inf  " if r["psnr"] == float("inf") else f"{r['psnr']:9.3f}"
        kvr = f"{r['kv_mem_reduction']:.2f}x" if r["kv_mem_reduction"] else "  -  "
        er = f"{r['e2e_mem_reduction']:.2f}x" if r["e2e_mem_reduction"] else "  -  "
        tot = f"{r['total_kv_mb']:12.2f}" if r["total_kv_mb"] else f"{'-':>12}"
        per = f"{r['per_layer_kv_mb']:12.2f}" if r["per_layer_kv_mb"] else f"{'-':>12}"
        peak = f"{r['peak_mb']:12.2f}" if r["peak_mb"] else f"{'-':>12}"
        print(f"{r['run']:6} | {psnr_s:>9} | {r['ssim']:7.4f} | {r['lpips']:7.4f} | "
              f"{per} | {tot} | {kvr:>6} | {peak} | {er:>6}")
    print("=" * 108)
    (BASE / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {BASE/'metrics.json'}")


if __name__ == "__main__":
    main()
