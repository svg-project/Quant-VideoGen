from functools import lru_cache
import torch
from enum import Enum
import re
import math
from .sim.quant.lowbit_quantize import (
    nvfp4_e2m1_quantize_triton,
    blockwise_int4_quantize_triton,
    blockwise_int3_quantize_triton,
    blockwise_int2_quantize_triton,
)
from .sim.quant.quantize_config import QuantizeConfig

from .functions import (
    kmeans_quantize_tensor,
    prq_quantize_tensor,
    triton_prq_quantize_tensor,
)


########################################################
# Entrypoints
########################################################

# Make it a enum
class QuantizeFunctions(Enum):
    NAIVE = "naive"
    KMEANS = "kmeans"
    NSTAGE_KMEANS = "prq"
    NSTAGE_KMEANS_CLIP = "prq_clip"
    TRITON_PRQ = "triton_prq"
    TRITON_PRQ_CLIP = "triton_prq_clip"


def get_quantize_fn(quant_type: str, quant_config: QuantizeConfig):
    if quant_type in ["naive-fp4", "kmeans-fp4", "nstages-kmeans-fp4", "nstages-kmeans-fp4-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-fp4-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int4", "kmeans-int4", "nstages-kmeans-int4", "nstages-kmeans-int4-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int4-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int3", "kmeans-int3", "nstages-kmeans-int3", "nstages-kmeans-int3-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int3-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int2", "kmeans-int2", "nstages-kmeans-int2", "nstages-kmeans-int2-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int2-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )
    elif quant_type in ["triton-nstages-kmeans-int2", "triton-nstages-kmeans-int2-clip", "triton-nstages-kmeans-int4", "triton-nstages-kmeans-int4-clip"]:
        """Do not quantize here"""
        def quantize_fn(x):
            m = re.search(r'int(\d+)', quant_config.quant_type)
            if m is None:
                raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
            num_bits = int(m.group(1))
            return num_bits
        
    else:
        raise ValueError(
            f"Unsupported quant type: {quant_type}"
        )
        
    return quantize_fn


def get_quantize_type(quant_type: str):
    # ==========================================================
    # Determine preprocessing mode
    # ==========================================================
    if quant_type in [
        "kmeans-fp4",
        "kmeans-fp4-clip",
        "kmeans-int4",
        "kmeans-int4-clip",
        "kmeans-int3",
        "kmeans-int3-clip",
        "kmeans-int2",
        "kmeans-int2-clip",
    ]:
        quantize_type = QuantizeFunctions.KMEANS

    elif quant_type in [
        "nstages-kmeans-fp4",
        "nstages-kmeans-int4",
        "nstages-kmeans-int3",
        "nstages-kmeans-int2",
    ]:
        quantize_type = QuantizeFunctions.NSTAGE_KMEANS

    elif quant_type in [
        "nstages-kmeans-fp4-clip",
        "nstages-kmeans-int4-clip",
        "nstages-kmeans-int3-clip",
        "nstages-kmeans-int2-clip",
    ]:
        quantize_type = QuantizeFunctions.NSTAGE_KMEANS_CLIP
    elif quant_type in [
        "triton-nstages-kmeans-int2",
        "triton-nstages-kmeans-int4",
    ]:
        quantize_type = QuantizeFunctions.TRITON_PRQ
    elif quant_type in [
        "triton-nstages-kmeans-int2-clip",
        "triton-nstages-kmeans-int4-clip",
    ]:
        quantize_type = QuantizeFunctions.TRITON_PRQ_CLIP
    else:
        quantize_type = QuantizeFunctions.NAIVE

    return quantize_type


def compress_kv_cache(k: torch.Tensor, v: torch.Tensor, quant_type: str, quant_config: QuantizeConfig, quantize_fn: callable):
    quantize_type = get_quantize_type(quant_type)

    if quantize_type == QuantizeFunctions.NSTAGE_KMEANS:
        # Apply PRQ (multi-stage K-Means) based quantization
        k_quant = prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_k_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
        v_quant = prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_v_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.NSTAGE_KMEANS_CLIP:

        # Apply PRQ (multi-stage K-Means) based quantization
        k_quant = prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_k_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
        v_quant = prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            codebook_size=quant_config.cache_num_v_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
    elif quantize_type == QuantizeFunctions.KMEANS:
        # Apply K-Means based quantization
        k_quant = kmeans_quantize_tensor(
            k,
            num_centroids=quant_config.cache_num_k_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
        v_quant = kmeans_quantize_tensor(
            v,
            num_centroids=quant_config.cache_num_v_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.TRITON_PRQ:
        # Apply Triton N-Stage K-Means based quantization
        asymmetric = getattr(quant_config, "asymmetric", False)
        k_quant = triton_prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_k_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            asymmetric=asymmetric,
        )
        v_quant = triton_prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_v_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            asymmetric=asymmetric,
        )
    elif quantize_type == QuantizeFunctions.TRITON_PRQ_CLIP:
        # Apply Triton N-Stage K-Means based quantization
        asymmetric = getattr(quant_config, "asymmetric", False)
        k_quant = triton_prq_quantize_tensor(
            k,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_k_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
            asymmetric=asymmetric,
        )
        v_quant = triton_prq_quantize_tensor(
            v,
            num_stages=quant_config.num_prq_stages,
            num_clusters=quant_config.cache_num_v_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
            asymmetric=asymmetric,
        )
    elif quantize_type == QuantizeFunctions.NAIVE:
        # ==========================================================
        # Direct quantization (no preprocessing)
        # ==========================================================
        k_quant = quantize_fn(k)
        v_quant = quantize_fn(v)
    else:
        raise ValueError(f"Unsupported quant type: {quant_type}")

    return k_quant, v_quant