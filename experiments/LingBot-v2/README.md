# LingBot-World-v2 KV-cache quantization

Integration of QuantVideoGen KV-cache quantization into LingBot-World-v2 (14B
causal DMD world model). One chunk = 3 latent frames = 12 pixel frames @ 16 fps,
so each chunk carries 0.75 s of video and the KV cache is rewritten continuously
while the stream runs.

```bash
bash scripts/LingBot-v2/download_models.sh   # -> ckpts/LingBot-v2/
bash scripts/LingBot-v2/run_bf16.sh          # bf16 baseline
bash scripts/LingBot-v2/run_qvg.sh           # quantized; edit the config block on top
```

Both launchers use the same example, seed, and attention window, so their outputs
are directly comparable. `--local_attn_size` counts the sink, so the scripts'
`68 = 62 rolling + 6 sink` latent frames give ~17 s of context.
`make_wasd_example.py` rebuilds the WASD example from a source frame.

## Quantization policy knobs

| Flag | Default | What it does |
|---|---|---|
| `--quant_type` | `none` | `triton-nstages-kmeans-int2` / `-int4` |
| `--cache_num_{k,v}_centroids` | 256 | k-means codebook size per span |
| `--num_prq_stages` | 1 | k-means stages before residual quantization |
| `--quant_block_size` | 64 | residual scale block size |
| `--kmeans_max_iters` | 2 | k-means iterations per stage |
| `--asymmetric` | off | KIVI-style min-max + zero point instead of absmax |
| `--quant_factor` | 8 | quantize the previous N chunks every N chunks (1 = per chunk) |
| `--quant_keep_recent_chunks` | 0 | newest completed chunks held in bf16 |
| `--quant_sink_keep_chunks` | 0 | leading sink chunks never quantized |

## Findings

Measured on the SGLang port of the same policy (`multimodal_gen` runtime, 8×H200
Ulysses, 16 videos, PSNR against a bf16 baseline generated in the same
environment). The knobs are identical, so the conclusions carry over, but the
numbers are not a like-for-like prediction of this path: the sweep ran with a
33+9 window over 3-latent chunks, while these scripts default to 68+6 over
4-latent chunks.

**Recommended setting.** int4 (quality) or int2 (compression), codebook 256,
block 64, 1 stage, 2 k-means iters, symmetric, `quant_factor 1` with
`--quant_keep_recent_chunks 1 --quant_sink_keep_chunks 1`.

1. **Residual bit width is the only large lever.** int4 vs int2 is ~9 dB of
   2–5 s PSNR (≈33.5 vs ≈24.5). Everything else below moves ≤3 dB.
2. **Keep the first sink chunk in bf16** (`--quant_sink_keep_chunks 1`). It holds
   the initial conditioning frame that every later chunk attends to forever, so
   its error never decays: keeping it gains **+15 dB (0–2 s)** and **+3 dB
   (2–5 s)** for ~9% of the whole-cache compression ratio. Best
   quality-per-byte trade in the sweep.
3. **Keep the newest completed chunk in bf16** (`--quant_keep_recent_chunks 1`).
   The chunk being denoised should see clean immediate history; this removes the
   early-window instability that per-chunk quantization otherwise introduces.
4. **Codebook size is cheap for int4, not for int2.** int4 c256→c64 costs ~1 dB
   while raising the span ratio 3.16×→3.63×; int2 loses ~2.3 dB over the same
   range (most of it at c128→c64), so int2 wants c256.
5. **k-means iterations barely matter** (1/2/4 within ~1 dB) *once the k-means
   RNG is seeded*, and iters=1 is only ~6% faster than 2. Use 2.
6. **Symmetric and asymmetric tie.** Block64-symmetric vs block128-asymmetric
   (equal metadata overhead) differ by ≤1 dB with no stable direction across 8
   paired configs, even though asymmetric wins ~30% on kernel-level RMSE for
   Gaussian data. Symmetric stays the default.
7. **Seed the k-means RNG, but isolate it.** `_quantize_kv_cache` wraps seeding
   in `torch.random.fork_rng`: seeding in place also resets the generation noise
   sequence, which pushed quantized runs off the bf16 trajectory and cost ~9 dB
   of *spurious* PSNR — enough to fabricate "collapses" (int4 at iters=1, small
   codebooks) that disappear once the RNG is isolated.

## Evaluating

`compute_metrics.py` computes PSNR/SSIM/LPIPS from the written `video.mp4`
(not float tensors), so codec noise is included on both sides.

Two measurement caveats, both learned the hard way:

- **Full-clip PSNR over 60 s is meaningless here.** Trajectories diverge
  chaotically: past ~20 s every configuration sits on a ~13 dB floor. Use
  time-windowed PSNR and read the **0–2 s / 2–5 s** windows; 5–10 s is
  directional; beyond that is noise.
- **Compare against a baseline generated in the same environment.** bf16 runs
  are bit-reproducible, but different launch environments are not, and quantized
  single-run differences within ±1–2 dB are not meaningful — average over
  several videos.
