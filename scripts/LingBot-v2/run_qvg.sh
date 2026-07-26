#!/bin/bash

example_dir=experiments/LingBot-v2/examples/shiba_wasd_1min
prompt=$(cat ${example_dir}/prompt.txt)
ckpt_path=ckpts/LingBot-v2/lingbot-world-v2-14b-causal-fast
frame_num=973          # 973 frames @ 16fps = ~60s
seed=42

#########################################################
# Sliding-window attention (latent frames; sink included)
#########################################################
local_attn_size=68     # ~17s of context
sink_size=6

#########################################################
# Quantization Configuration
#########################################################
quant_type="triton-nstages-kmeans-int4"
# quant_type="triton-nstages-kmeans-int2"
cache_num_k_centroids=256
cache_num_v_centroids=256
kmeans_max_iters=2
quant_block_size=64
num_prq_stages=1

#########################################################
# Which chunks get quantized (see experiments/LingBot-v2/README.md)
#########################################################
quant_factor=1              # 1 = quantize every chunk as it completes
keep_recent_chunks=1        # newest completed chunk stays bf16
sink_keep_chunks=1          # first sink chunk (conditioning frame) stays bf16

quant_dir=${quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
output_folder=results/lingbot_v2/${quant_dir}

echo "Running inference with checkpoint $ckpt_path and example ${example_dir}"
echo "Output will be saved to $output_folder"

export PYTHONPATH=experiments/LingBot-v2:.

torchrun --nproc_per_node=8 --standalone experiments/LingBot-v2/generate.py \
  --task i2v-A14B \
  --size 720*1280 \
  --ckpt_dir $ckpt_path \
  --image ${example_dir}/image.jpg \
  --action_path ${example_dir} \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --frame_num $frame_num \
  --local_attn_size $local_attn_size \
  --sink_size $sink_size \
  --base_seed $seed \
  --save_dir $output_folder \
  --save_file $output_folder/video.mp4 \
  --prompt "$prompt" \
  --use_chunked_kv \
  --quant_type $quant_type \
  --cache_num_k_centroids $cache_num_k_centroids \
  --cache_num_v_centroids $cache_num_v_centroids \
  --kmeans_max_iters $kmeans_max_iters \
  --quant_block_size $quant_block_size \
  --num_prq_stages $num_prq_stages \
  --quant_factor $quant_factor \
  --quant_keep_recent_chunks $keep_recent_chunks \
  --quant_sink_keep_chunks $sink_keep_chunks
