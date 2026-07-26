#!/bin/bash

# Download LingBot-World-v2 14B causal-fast: the streaming/causal DiT used by
# experiments/LingBot-v2/generate.py, together with the VAE and T5 encoder.
hf download robbyant/lingbot-world-v2-14b-causal-fast \
  --local-dir ckpts/LingBot-v2/lingbot-world-v2-14b-causal-fast
