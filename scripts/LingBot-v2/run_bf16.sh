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

output_folder=results/lingbot_v2/bf16

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
  --prompt "$prompt"
