import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool


EXAMPLE_PROMPT = {
    "i2v-A14B": {
        "prompt":
            "A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.",
        "image":
            "examples/04/image.jpg",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]

    if args.task == "i2v-A14B":
        assert args.image is not None, "Please specify the image path for i2v."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    if not 0 <= args.video_crf <= 51:
        raise ValueError(f"video_crf must be between 0 and 51, got {args.video_crf}")
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="causal_fast",
        choices=["causal_fast", "causal_pretrain"],
        help="Inference mode: 'causal_fast' for the distilled few-step model, "
             "'causal_pretrain' for the pretrained causal model with 40-step CFG sampling.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="The chunk size."),
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--video_codec",
        type=str,
        default="libx264",
        choices=["libx264", "libx265"],
        help="Video encoder. libx264 is broadly compatible; libx265 is smaller.")
    parser.add_argument(
        "--video_crf",
        type=int,
        default=23,
        help="Constant-quality level from 0 (lossless) to 51 (lowest quality). "
             "23 is near-transparent for ~480p output at ~3x smaller than 18.")
    parser.add_argument(
        "--video_preset",
        type=str,
        default="slow",
        choices=[
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ],
        help="Encoder speed/compression tradeoff.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="The camera path to generate the video from.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
    parser.add_argument(
        "--local_attn_size",
        type=int,
        default=-1,
        help='The local size of kv cache during inference')
    parser.add_argument(
        "--sink_size",
        type=int,
        default=0,
        help='The sink size of kv cache during inference')
    parser.add_argument(
        "--max_attention_size",
        type=int,
        default=None,
        help="The size of kv cache during inference.")
    # ---- QuantVideoGen KV-cache quantization ----
    parser.add_argument(
        "--use_chunked_kv", action="store_true", default=False,
        help="Use QVG ChunkedKVCache for self-attn KV storage. Required when "
             "--quant_type is set. Run with --local_attn_size -1 (global cache).")
    parser.add_argument(
        "--quant_type", type=str, default="none",
        help="KV-cache quant type. 'none' disables. e.g. "
             "triton-nstages-kmeans-int4, triton-nstages-kmeans-int2.")
    # Defaults below follow the 36-config x 16-video sweep in README.md:
    # 1 stage / block 64 / 2 k-means iters, and centroids matter far less than
    # the residual bit width (int4 tolerates c64, int2 prefers c256).
    parser.add_argument("--cache_num_k_centroids", type=int, default=256)
    parser.add_argument("--cache_num_v_centroids", type=int, default=256)
    parser.add_argument(
        "--kmeans_max_iters", type=int, default=2,
        help="K-means iterations per stage. 1/2/4 differ by <1 dB PSNR once the "
             "k-means RNG is seeded; 2 is the safe cheap default.")
    parser.add_argument("--quant_block_size", type=int, default=64)
    parser.add_argument(
        "--num_prq_stages", type=int, default=1,
        help="PRQ k-means stages. 1 is enough for per-chunk quantization: extra "
             "stages add codebook overhead per span without a quality win.")
    parser.add_argument(
        "--asymmetric", action="store_true", default=False,
        help="KIVI-style asymmetric residual quantization (per-block min-max + "
             "zero point) instead of symmetric absmax. Symmetric int2 only uses "
             "{-1,0,1}; asymmetric spans the full unsigned range but needs a "
             "zero point per block. Ties with symmetric end-to-end at equal "
             "overhead (block64-symm vs block128-asym), so it is off by default.")
    parser.add_argument(
        "--quant_factor", type=int, default=8,
        help="Quantize the previous `quant_factor` chunks once every "
             "`quant_factor` chunks (Self-Forcing QUANT_FACTOR convention). "
             "Use 1 for per-chunk quantization plus --quant_keep_recent_chunks.")
    parser.add_argument(
        "--quant_keep_recent_chunks", type=int, default=0,
        help="Completed chunks kept in BF16 before quantization reaches them, so "
             "the chunk being denoised sees clean recent history. 1 recommended.")
    parser.add_argument(
        "--quant_sink_keep_chunks", type=int, default=0,
        help="Leading attention-sink chunks never quantized. The first sink chunk "
             "holds the initial conditioning frame every later chunk attends to "
             "forever: keeping it BF16 buys ~+15 dB (0-2s) / ~+3 dB (2-5s) PSNR "
             "for ~9%% of the whole-cache ratio. 1 recommended.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='output',
        help="The path to the checkpoint directory.")

    args = parser.parse_args()
    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def run_causal(args, cfg, img, device, rank, mode="causal_fast"):
    """Run inference with the distilled few-step model (infer_mode='causal_fast')."""
    logging.info(f"Creating WanI2VCausal pipeline (infer_mode={mode}).")
    from types import SimpleNamespace
    quant_config = SimpleNamespace(
        quant_type=args.quant_type,
        cache_num_k_centroids=args.cache_num_k_centroids,
        cache_num_v_centroids=args.cache_num_v_centroids,
        kmeans_max_iters=args.kmeans_max_iters,
        quant_block_size=args.quant_block_size,
        num_prq_stages=args.num_prq_stages,
        asymmetric=args.asymmetric,
        quant_factor=args.quant_factor,
        quant_keep_recent_chunks=args.quant_keep_recent_chunks,
        quant_sink_keep_chunks=args.quant_sink_keep_chunks,
    )
    use_chunked_kv = args.use_chunked_kv or args.quant_type != "none"
    wan_i2v = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode=mode,
        use_chunked_kv=use_chunked_kv,
        quant_config=quant_config,
    )
    logging.info("Generating video ...")
    return wan_i2v.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    cfg = WAN_CONFIGS[args.task]

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), "sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        init_distributed_group()

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    video = run_causal(args, cfg, img, device, rank, mode=args.infer_mode)

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    logging.info(f"[END-TO-END] Peak GPU memory (rank {rank}): {peak_mem_mb:.2f} MB")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "peak_mem_mb.txt"), "w") as f:
            f.write(f"{peak_mem_mb:.2f}\n")
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = '.mp4'
            args.save_file = f"lingbot-world-v2-{args.infer_mode}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
            args.save_file = f'{args.save_dir}/{args.save_file}'

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
            codec=args.video_codec,
            crf=args.video_crf,
            preset=args.video_preset)

    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
