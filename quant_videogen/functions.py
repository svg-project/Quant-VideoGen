import torch
import math
from functools import lru_cache

from .kmeans.kmeans_euclid import batch_kmeans_Euclid

from .kernels.triton.permute import (
    permute_tensor_by_labels_triton,
    apply_inverse_permutation_triton,
)
from .real.prq import (
    prq_quant,
    prq_dequant,
)

from .sim.quant.lowbit_quantize import (
    compute_percentile_by_sorting,
    percentile_clip_and_scale,
    percentile_unscale_and_add_residual,
)

def subtract_cluster_centroids(
    tensor: torch.Tensor,
    centroids: torch.Tensor,
    cluster_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Subtract cluster centroids from a permuted tensor.

    Args:
        tensor: Permuted tensor of shape [B, H, S, D]
        centroids: Cluster centroids of shape [B, H, num_centroids, D]
        cluster_sizes: Size of each cluster [B, H, num_centroids]

    Returns:
        Tensor with centroids subtracted, same shape as input
    """
    B, H, S, D = tensor.shape
    num_centroids = centroids.shape[2]
    result = tensor.clone()

    for b in range(B):
        for h in range(H):
            start = 0
            for c in range(num_centroids):
                c_size = cluster_sizes[b, h, c].item()
                if c_size > 0:
                    result[b, h, start : start + c_size] -= centroids[b, h, c]
                    start += c_size

    return result


def add_cluster_centroids(
    tensor: torch.Tensor,
    centroids: torch.Tensor,
    cluster_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Add cluster centroids back to a permuted tensor.

    Args:
        tensor: Permuted tensor of shape [B, H, S, D]
        centroids: Cluster centroids of shape [B, H, num_centroids, D]
        cluster_sizes: Size of each cluster [B, H, num_centroids]

    Returns:
        Tensor with centroids added, same shape as input
    """
    B, H, S, D = tensor.shape
    num_centroids = centroids.shape[2]
    result = tensor.clone()

    for b in range(B):
        for h in range(H):
            start = 0
            for c in range(num_centroids):
                c_size = cluster_sizes[b, h, c].item()
                if c_size > 0:
                    result[b, h, start : start + c_size] += centroids[b, h, c]
                    start += c_size

    return result


def kmeans_quantize_tensor(
    tensor: torch.Tensor,
    num_centroids: int,
    kmeans_max_iters: int,
    quantize_fn,
) -> torch.Tensor:
    """
    Apply K-Means based quantization to a tensor.

    The process:
    1. Cluster the tensor using K-Means
    2. Permute by cluster labels
    3. Subtract cluster centroids
    4. Apply quantization
    5. Add centroids back
    6. Inverse permute to restore original order

    Args:
        tensor: Input tensor of shape [B, H, S, D]
        num_centroids: Number of K-Means clusters
        kmeans_max_iters: Maximum iterations for K-Means
        quantize_fn: Quantization function to apply

    Returns:
        Quantized tensor with same shape as input
    """
    B, H, S, D = tensor.shape

    # Reshape for batch K-Means: [B*H, S, D]
    tensor_reshaped = tensor.view(B * H, S, D)

    # Run K-Means
    labels, centroids, cluster_sizes, _ = batch_kmeans_Euclid(
        tensor_reshaped,
        n_clusters=num_centroids,
        max_iters=kmeans_max_iters,
    )

    # Reshape labels: [B, H, S]
    labels = labels.view(B, H, S)

    # Permute by cluster labels
    permuted, sorted_indices = permute_tensor_by_labels_triton(tensor, labels, dim=2)

    # Reshape cluster sizes and centroids
    cluster_sizes = cluster_sizes.view(B, H, num_centroids)
    centroids = centroids.view(B, H, num_centroids, D)

    # Subtract cluster centroids
    to_quantize = subtract_cluster_centroids(permuted, centroids, cluster_sizes)

    # Apply quantization
    quantized = quantize_fn(to_quantize)

    # Add centroids back
    perm_quant = add_cluster_centroids(quantized, centroids, cluster_sizes)

    # Apply inverse permutation to restore original order
    result = apply_inverse_permutation_triton(perm_quant, sorted_indices, dim=2)

    return result


def prq_quantize_tensor(
    tensor: torch.Tensor,
    num_stages: int,
    codebook_size: int,
    kmeans_max_iters: int,
    quantize_fn=None,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
) -> torch.Tensor:
    """
    Simulation of the Progresive Residual Quantization (PRQ) algorithm.

    The process:
    1. Run k-means to get centroids and cluster assignments
    2. Subtract gathered centroids to get residual
    3. Repeat on residual for num_stages iterations
    4. Reconstruct by summing all gathered centroids + quantized residual

    Args:
        tensor: Input tensor of shape [B, H, S, D]
        num_stages: Number of PRQ stages (k-means runs)
        codebook_size: Number of centroids per stage
        kmeans_max_iters: Maximum iterations for K-Means
        quantize_fn: Optional quantization function for final residual.
                     If None, only PRQ reconstruction is returned.
        use_percentile_clipping: If True, clip values based on percentile threshold
                                 to avoid extreme outliers affecting quantization
        percentile: The percentile threshold for clipping (default: 99.0 for top 1%)

    Returns:
        Quantized tensor with same shape as input
    """
    B, H, S, D = tensor.shape

    # Percentile-based clipping and scaling to handle extreme outliers
    pclip_residual = None
    pclip_scale_factor = 1.0
    if use_percentile_clipping:
        tensor_abs = tensor.abs()
        # Get the value at the specified percentile (e.g., top 1% = 99th percentile)
        percentile_value = compute_percentile_by_sorting(tensor_abs, percentile)

        if percentile_value > 0:
            # Clip values to the percentile threshold
            tensor_clipped = torch.clamp(
                tensor, min=-percentile_value, max=percentile_value
            )

            # Get the residual (values that were clipped off)
            clipped_mask = tensor_clipped != tensor
            pclip_residual = tensor * clipped_mask
            tensor_clipped = tensor - pclip_residual

            # Scale so that the clipped tensor uses more dynamic range
            pclip_scale_factor = 1.0 / percentile_value
            tensor = tensor_clipped * pclip_scale_factor

    # Reshape for batch K-Means: [B*H, S, D]
    tensor_reshaped = tensor.view(B * H, S, D).contiguous()

    # PRQ: multi-stage k-means
    residual = tensor_reshaped
    indices_list = []
    centers_list = []

    for _ in range(num_stages):
        # Run K-Means on current residual
        labels, centroids, _, _ = batch_kmeans_Euclid(
            residual,
            n_clusters=codebook_size,
            max_iters=kmeans_max_iters,
        )
        labels = labels.long()
        indices_list.append(labels)
        centers_list.append(centroids)

        # Gather the reconstruction for this stage: [B*H, S, D]
        gathered = torch.gather(
            centroids,
            dim=1,
            index=labels.unsqueeze(-1).expand(-1, -1, D),
        )
        # Update residual
        residual = residual - gathered

    # Quantize final residual if quantize_fn provided
    if quantize_fn is not None:
        # Reshape residual to [B, H, S, D] for quantize_fn
        residual = residual.view(B, H, S, D)
        residual_quantized = quantize_fn(residual)
        residual = residual_quantized.view(B * H, S, D)

    # Reconstruct: sum of all gathered centroids + (quantized) residual
    reconstruction = residual
    for labels, centroids in zip(indices_list, centers_list):
        gathered = torch.gather(
            centroids,
            dim=1,
            index=labels.unsqueeze(-1).expand(-1, -1, D),
        )
        reconstruction = reconstruction + gathered

    # Reshape back to [B, H, S, D]
    result = reconstruction.view(B, H, S, D)

    # Scale back and add residual from percentile clipping
    if use_percentile_clipping and pclip_residual is not None:
        result = result / pclip_scale_factor + pclip_residual

    return result


def triton_prq_quantize_tensor(
    tensor: torch.Tensor,
    num_stages: int,
    num_clusters: int,
    block_size: int,
    max_iters: int = 100,
    scale_precision: torch.dtype = torch.float8_e4m3fn,
    use_percentile_clipping: bool = False,
    percentile: float = 99.0,
    quantize_fn=None,
    asymmetric: bool = False,
) -> dict:
    """
    Apply Triton-based N-stage K-Means quantization to a tensor.

    This uses optimized Triton kernels for multi-stage K-Means quantization.
    Returns a dictionary containing all quantization components for later
    dequantization.

    Args:
        tensor: Input tensor of shape [B, H, S, D]
        num_stages: Number of K-Means stages
        num_clusters: Number of centroids per stage
        block_size: Block size for residual quantization
        num_bits: Number of bits for residual quantization (2 or 4)
        max_iters: Maximum iterations for K-Means
        scale_precision: Precision for scale factors
        use_percentile_clipping: If True, apply percentile clipping before quantization
        percentile: Percentile threshold for clipping (default: 99.0)

    Returns:
        Dictionary containing:
            - centroids_list: List of centroids for each stage
            - cluster_ids_list: List of cluster assignments for each stage
            - residual_quant: Quantized residual tensor
            - scales: Scale factors for residual
            - residual: (if clipping) Clipped residual
            - scale_factor: (if clipping) Scale factor used
            - quant_type: Quantization type string
            - num_bits: Number of bits used
    """
    num_bits = quantize_fn(tensor)
    if num_bits == 2:
        TARGET_MAX = 448 * 1
    elif num_bits == 4:
        TARGET_MAX = 448 * 7
    else:
        raise ValueError(f"Unsupported num_bits: {num_bits}")
    
    if use_percentile_clipping:
        tensor, residual, scale_factor = percentile_clip_and_scale(
            tensor,
            percentile=percentile,
            target_max=TARGET_MAX,
        )

    centroids_list, cluster_ids_list, residual_quant, scales, zeros = prq_quant(
        tensor.contiguous(),
        n_stages=num_stages,
        n_clusters=num_clusters,
        block_size=block_size,
        num_bits=num_bits,
        scale_precision=scale_precision,
        max_iters=max_iters,
        PACK_OUTPUT_INT8=True,
        CLUSTER_ID_INT8=True,
        asymmetric=asymmetric,
    )

    if use_percentile_clipping:
        return {
            "centroids_list": centroids_list,
            "cluster_ids_list": cluster_ids_list,
            "residual_quant": residual_quant,
            "scales": scales,
            "zeros": zeros,
            "residual": residual,
            "scale_factor": scale_factor,
        }
    else:
        return {
            "centroids_list": centroids_list,
            "cluster_ids_list": cluster_ids_list,
            "residual_quant": residual_quant,
            "scales": scales,
            "zeros": zeros,
        }

    # dequantized_tensor = prq_dequant(
    #     centroids_list=centroids_list,
    #     cluster_ids_list=cluster_ids_list,
    #     residual_quant=residual_quant,
    #     scales=scales,
    #     block_size=block_size,
    #     num_bits=num_bits,
    #     PACK_INPUT_INT8=True,
    #     CLUSTER_ID_INT8=True,
    #     output_dtype=tensor.dtype,
    # )
    # if use_percentile_clipping:
    #     dequantized_tensor = percentile_unscale_and_add_residual(dequantized_tensor, residual, scale_factor)
    
    # return dequantized_tensor


def triton_prq_dequantize_tensor(
    packed_state: dict,
    block_size: int,
    num_bits: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Reconstruct tensor from packed PRQ state produced by triton_prq_quantize_tensor.

    Args:
        packed_state: Dict with keys:
            - centroids_list: List of tensors from each stage
            - cluster_ids_list: List of tensors from each stage
            - residual_quant: Quantized residual tensor
            - scales: Scale factors
            - residual (optional): If use_percentile_clipping was used
            - scale_factor (optional): If use_percentile_clipping was used
            - use_percentile_clipping (optional): bool
        block_size: Block size used in quantization.
        num_bits: Number of bits (2 or 4) used in quantization.
        output_dtype: Dtype of the reconstructed tensor.

    Returns:
        Reconstructed tensor of shape [B, H, S, D].
    """
    centroids_list = packed_state["centroids_list"]
    cluster_ids_list = packed_state["cluster_ids_list"]
    residual_quant = packed_state["residual_quant"]
    scales = packed_state["scales"]

    dequantized_tensor = prq_dequant(
        centroids_list=centroids_list,
        cluster_ids_list=cluster_ids_list,
        residual_quant=residual_quant,
        scales=scales,
        block_size=block_size,
        num_bits=num_bits,
        PACK_INPUT_INT8=True,
        CLUSTER_ID_INT8=True,
        output_dtype=output_dtype,
        zeros=packed_state.get("zeros"),
    )

    if packed_state.get("residual", None) is not None:
        residual = packed_state["residual"]
        scale_factor = packed_state["scale_factor"]

        dequantized_tensor = percentile_unscale_and_add_residual(
            dequantized_tensor,
            residual,
            scale_factor,
        )

    return dequantized_tensor
