"""
Chunk-based KV cache: manages a *single* K or V tensor in frame-aligned
chunks, supporting mixed BF16 / quantized storage.
"""

from enum import IntEnum
from typing import Optional
import torch

from .uncompress import uncompress_kv_cache

class ChunkState(IntEnum):
    EMPTY = 0
    BF16 = 1
    QUANTIZED = 2



class ChunkedKVCache:
    """
    Manages **one** tensor (either K or V) in frame-aligned chunks.

    * BF16 chunks are stored individually; memory is allocated lazily with
      ``torch.empty`` on first write.
    * Real-quantized spans (dicts from ``compress_kv_cache``) can cover
      multiple contiguous chunks and are stored as a single compressed object.
    * ``read()`` returns a full-precision tensor in the configured *layout*,
      decompressing quantized spans on the fly.

    Args:
        layout: ``"BHSD"`` stores chunks as ``[B, H, S, D]`` (default);
                ``"BSHD"`` stores chunks as ``[B, S, H, D]``.
                All ``write`` / ``read`` / ``store_quantized`` calls use
                the same layout — no implicit permutes.
    """

    def __init__(
        self,
        batch_size: int,
        frame_seq_length: int,
        num_heads: int,
        head_dim: int,
        max_num_chunks: int,
        dtype: torch.dtype,
        device: torch.device,
        layout: str = "BHSD",
    ):
        assert layout in ("BHSD", "BSHD"), f"layout must be BHSD or BSHD, got {layout}"
        self.batch_size = batch_size
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_chunks = max_num_chunks
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.seq_dim = 2 if layout == "BHSD" else 1

        # Per-chunk BF16 storage (None → not yet allocated, zero GPU memory)
        self.chunks: list[Optional[torch.Tensor]] = [None] * max_num_chunks
        self.chunk_state: list[ChunkState] = [ChunkState.EMPTY] * max_num_chunks

        # Quantized spans: {start_chunk, end_chunk, quant_data}
        # quant_data is the packed dict from compress_kv_cache (with "info").
        self.quantized_spans: list[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kv_cache_size(self) -> int:
        """Total token capacity."""
        return self.max_num_chunks * self.frame_seq_length

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _token_to_chunk(self, token_index: int) -> int:
        assert token_index % self.frame_seq_length == 0, (
            f"Token index {token_index} not aligned to "
            f"frame_seq_length {self.frame_seq_length}"
        )
        return token_index // self.frame_seq_length

    def _alloc_chunk(self, ci: int):
        if self.chunks[ci] is None:
            if self.layout == "BHSD":
                shape = [self.batch_size, self.num_heads, self.frame_seq_length, self.head_dim]
            else:
                shape = [self.batch_size, self.frame_seq_length, self.num_heads, self.head_dim]
            self.chunks[ci] = torch.empty(shape, dtype=self.dtype, device=self.device)

    def _find_span(self, ci: int) -> dict:
        for span in self.quantized_spans:
            if span["start_chunk"] <= ci < span["end_chunk"]:
                return span
        raise ValueError(f"No quantized span covers chunk {ci}")

    def _remove_overlapping_spans(self, start_chunk: int, end_chunk: int):
        self.quantized_spans = [
            s for s in self.quantized_spans
            if s["end_chunk"] <= start_chunk or s["start_chunk"] >= end_chunk
        ]

    # ------------------------------------------------------------------
    # Write (BF16)
    # ------------------------------------------------------------------

    def write(self, start_index: int, end_index: int, data: torch.Tensor):
        """
        Write BF16 data into the cache.

        Args:
            start_index: Token position (frame-aligned).
            end_index: End token position (frame-aligned).
            data: Tensor in the cache's configured layout.
                  BHSD → ``[B, H, num_tokens, D]``;
                  BSHD → ``[B, num_tokens, H, D]``.
        """
        num_tokens = data.shape[self.seq_dim]
        assert end_index - start_index == num_tokens, (
            f"end_index - start_index ({end_index - start_index}) != "
            f"data tokens ({num_tokens})"
        )
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)
        assert ec <= self.max_num_chunks, (
            f"Write exceeds capacity: chunk {ec} > max {self.max_num_chunks}"
        )

        for i, ci in enumerate(range(sc, ec)):
            s = i * self.frame_seq_length
            e = s + self.frame_seq_length

            if self.chunk_state[ci] == ChunkState.QUANTIZED:
                self._remove_overlapping_spans(ci, ci + 1)

            self._alloc_chunk(ci)
            if self.seq_dim == 2:  # BHSD
                self.chunks[ci].copy_(data[:, :, s:e, :])
            else:  # BSHD
                self.chunks[ci].copy_(data[:, s:e, :, :])
            self.chunk_state[ci] = ChunkState.BF16

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, start_index: int, end_index: int) -> torch.Tensor:
        """
        Return full-precision tensor for ``[start_index, end_index)``
        in the cache's configured layout.  Decompresses quantized spans
        on the fly.
        """
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)
        parts: list[torch.Tensor] = []
        ci = sc

        while ci < ec:
            state = self.chunk_state[ci]

            if state == ChunkState.BF16:
                parts.append(self.chunks[ci])
                ci += 1

            elif state == ChunkState.QUANTIZED:
                span = self._find_span(ci)
                # uncompress always returns BHSD [B, H, S, D]
                dec, _ = uncompress_kv_cache(span["quant_data"], span["quant_data"])
                if self.layout == "BSHD":
                    dec = dec.permute(0, 2, 1, 3).contiguous()

                span_start = span["start_chunk"]
                span_end = span["end_chunk"]
                tok_off = (ci - span_start) * self.frame_seq_length
                tok_len = (min(span_end, ec) - ci) * self.frame_seq_length
                if self.seq_dim == 2:  # BHSD
                    parts.append(dec[:, :, tok_off:tok_off + tok_len, :])
                else:  # BSHD
                    parts.append(dec[:, tok_off:tok_off + tok_len, :, :])
                ci = min(span_end, ec)

            else:
                raise ValueError(f"Chunk {ci} is {ChunkState.EMPTY.name}, cannot read")

        return torch.cat(parts, dim=self.seq_dim)

    # ------------------------------------------------------------------
    # Store quantized data (output of compress_kv_cache, [B, H, S, D])
    # ------------------------------------------------------------------

    def store_quantized(self, start_index: int, end_index: int, quant_data):
        """
        Store quantized result for one tensor (K or V).

        * **Tensor** (fake quant): written in the cache's native layout.
        * **Dict** (real quant): stored as a single quantized span.
          Must already contain ``"info"`` with ``output_dtype`` /
          ``quant_config`` (via ``_pack_info_into_kv_cache``).
        """
        sc = self._token_to_chunk(start_index)
        ec = self._token_to_chunk(end_index)

        if isinstance(quant_data, torch.Tensor):
            self.write(start_index, end_index, quant_data)
        else:
            self._remove_overlapping_spans(sc, ec)
            for ci in range(sc, ec):
                self.chunks[ci] = None
                self.chunk_state[ci] = ChunkState.QUANTIZED
            self.quantized_spans.append({
                "start_chunk": sc,
                "end_chunk": ec,
                "quant_data": quant_data,
            })

    # ------------------------------------------------------------------
    # Sliding-window eviction
    # ------------------------------------------------------------------

    def evict_range(self, free_start_index: int, free_end_index: int):
        """Free storage for the token span ``[free_start_index, free_end_index)``
        that has slid out of the attention window.

        * BF16 chunks fully inside the freed span are released (set to ``None``).
        * A **quantized span** is dropped — releasing its centroids *and* its
          per-token data — **only if every chunk it covers lies inside the
          freed span**.  A span that still overlaps a live region (e.g. its
          most-recent frames are still in the window) is kept **intact**, so its
          shared centroids remain valid for the surviving tokens.  This is the
          "keep the centroid until all of its chunks are evicted" rule.

        No data is moved: chunks stay at their absolute positions, so quantized
        spans are never re-materialized to BF16 by eviction.
        """
        sc = self._token_to_chunk(free_start_index)
        ec = self._token_to_chunk(free_end_index)

        # Release BF16 chunks that fall entirely inside the freed span.
        for ci in range(sc, ec):
            if self.chunk_state[ci] == ChunkState.BF16:
                self.chunks[ci] = None
                self.chunk_state[ci] = ChunkState.EMPTY

        # Drop only the quantized spans that are fully contained in the freed
        # span; keep partially-live spans (centroids stay resident).
        survivors: list[dict] = []
        for span in self.quantized_spans:
            if sc <= span["start_chunk"] and span["end_chunk"] <= ec:
                for ci in range(span["start_chunk"], span["end_chunk"]):
                    self.chunks[ci] = None
                    self.chunk_state[ci] = ChunkState.EMPTY
            else:
                survivors.append(span)
        self.quantized_spans = survivors

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self):
        self.chunks = [None] * self.max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * self.max_num_chunks
        self.quantized_spans.clear()

    # ------------------------------------------------------------------
    # Offload / onload
    # ------------------------------------------------------------------

    def offload(self):
        for i in range(self.max_num_chunks):
            if self.chunks[i] is not None:
                self.chunks[i] = self.chunks[i].to("cpu")
        for span in self.quantized_spans:
            span["quant_data"] = _move_item(span["quant_data"], "cpu")

    def onload(self, device: torch.device):
        for i in range(self.max_num_chunks):
            if self.chunks[i] is not None:
                self.chunks[i] = self.chunks[i].to(device)
        for span in self.quantized_spans:
            span["quant_data"] = _move_item(span["quant_data"], device)


# ======================================================================
# Layer-dict helpers  (operate on the {"k":…, "v":…, …} dict)
# ======================================================================

def offload_kv_cache_layer(layer: dict):
    """Offload an entire layer dict (with ChunkedKVCache values) to CPU."""
    for key, val in layer.items():
        if isinstance(val, ChunkedKVCache):
            val.offload()
        elif isinstance(val, torch.Tensor):
            layer[key] = val.to("cpu")


def onload_kv_cache_layer(layer: dict, device: torch.device):
    """Onload an entire layer dict to *device*."""
    for key, val in layer.items():
        if isinstance(val, ChunkedKVCache):
            val.onload(device)
        elif isinstance(val, torch.Tensor):
            layer[key] = val.to(device)


# ======================================================================
# Internal helper
# ======================================================================

def _move_item(item, device):
    """Recursively move a tensor / dict-of-tensors to *device*."""
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        for key, val in item.items():
            if isinstance(val, torch.Tensor):
                item[key] = val.to(device)
            elif isinstance(val, list):
                item[key] = [
                    t.to(device) if isinstance(t, torch.Tensor) else t
                    for t in val
                ]
        return item
    return item
