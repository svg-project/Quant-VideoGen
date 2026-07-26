# LingBot-World-v2

QuantVideoGen KV-cache quantization for LingBot-World-v2 (14B causal DiT). The
model generates in chunks of latent frames and keeps a sliding attention window
with an attention sink, so the KV cache is written and re-read continuously
while the stream runs.

## Run

```bash
bash scripts/LingBot-v2/download_models.sh   # -> ckpts/LingBot-v2/
bash scripts/LingBot-v2/run_bf16.sh          # bf16 baseline
bash scripts/LingBot-v2/run_qvg.sh           # quantized; edit the config block on top
```

Both launchers use the same example, seed, and attention window, so their
outputs are directly comparable. `--local_attn_size` counts the sink, so the
scripts' `68 = 62 rolling + 6 sink` latent frames give ~17 s of context.
`make_wasd_example.py` rebuilds the WASD example from a source frame.

## Quantization flags

| Flag | Default | What it does |
|---|---|---|
| `--quant_type` | `none` | `triton-nstages-kmeans-int2` / `-int4` |
| `--cache_num_{k,v}_centroids` | 256 | k-means codebook size per span |
| `--num_prq_stages` | 1 | k-means stages before residual quantization |
| `--quant_block_size` | 64 | residual scale block size |
| `--kmeans_max_iters` | 2 | k-means iterations per stage |
| `--asymmetric` | off | min-max + zero point instead of symmetric absmax |
| `--quant_factor` | 8 | quantize the previous N chunks every N chunks (1 = per chunk) |
| `--quant_keep_recent_chunks` | 0 | newest completed chunks held in bf16 |
| `--quant_sink_keep_chunks` | 0 | leading sink chunks never quantized |

The last three decide *which* chunks get quantized rather than how. Two of them
are worth turning on: `--quant_keep_recent_chunks 1` leaves the chunk being
denoised with clean immediate history, and `--quant_sink_keep_chunks 1` keeps the
first sink chunk — it holds the initial conditioning frame that every later chunk
attends to, so quantization error there never decays. Both cost a little
compression ratio and buy back quality, most visibly early in the clip.

The residual bit width dominates everything else: pick int4 for quality or int2
for compression first, then tune the rest. Codebook size matters more at int2
than at int4.

## Evaluating

`compute_metrics.py` computes PSNR/SSIM/LPIPS from the written `video.mp4`, so
codec noise is included on both sides of a comparison.

Two caveats when comparing runs of a streaming model:

- **Full-clip PSNR over a long clip is not a useful signal.** Trajectories
  diverge chaotically, and late in the clip every configuration converges to the
  same low floor regardless of quantization error. Compare time-windowed PSNR and
  read the early windows.
- **Generate the baseline in the same environment as the run under test**, and
  average over several examples: single-run differences of a dB or two are not
  meaningful.
