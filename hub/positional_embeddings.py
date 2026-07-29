import torch
import torch.nn as nn


class RoPESplitHalf(nn.Module):
    """LLaMA-style split-half Rotary Position Embedding."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()

        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")

        if head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got {head_dim}"
            )

        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")

        self.head_dim = head_dim
        self.base = float(base)

        # Retain this buffer name for compatibility with your existing
        # checkpoint. Tables are still calculated in float32 below.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        # These are runtime caches, not model state.
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cached_len = 0
        self._cached_device: torch.device | None = None

    def _build_cache(
        self,
        seq_len: int,
        device: torch.device,
    ) -> None:
        # Recalculate frequencies in float32. This avoids unnecessary
        # positional precision loss when the model uses fp16/bfloat16.
        frequency_indices = torch.arange(
            0,
            self.head_dim,
            2,
            device=device,
            dtype=torch.float32,
        )

        inv_freq = 1.0 / (
            self.base ** (frequency_indices / self.head_dim)
        )

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=torch.float32,
        )

        frequencies = torch.outer(positions, inv_freq)

        # Split-half layout:
        # [f0, f1, ..., fN, f0, f1, ..., fN]
        embeddings = torch.cat(
            (frequencies, frequencies),
            dim=-1,
        )

        self._cos_cached = embeddings.cos()
        self._sin_cached = embeddings.sin()
        self._cached_len = seq_len
        self._cached_device = device

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE values for specified token positions.

        Args:
            x:
                Tensor used to determine device and output dtype.

            seq_len:
                Total sequence length required by the cache. During
                cached generation this is past_length + current_length.

            position_ids:
                Optional positions with shape (B, T). If omitted,
                positions 0 through seq_len - 1 are returned.

        Returns:
            If position_ids is provided:
                cos and sin with shape (B, T, head_dim).

            Otherwise:
                cos and sin with shape (seq_len, head_dim).
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        if (
            self._cos_cached is None
            or self._sin_cached is None
            or seq_len > self._cached_len
            or self._cached_device != x.device
        ):
            self._build_cache(seq_len, x.device)

        cos = self._cos_cached
        sin = self._sin_cached

        if position_ids is None:
            cos = cos[:seq_len]
            sin = sin[:seq_len]
        else:
            position_ids = position_ids.to(
                device=x.device,
                dtype=torch.long,
            )

            if position_ids.ndim != 2:
                raise ValueError(
                    "position_ids must have shape (B, T), "
                    f"got {position_ids.shape}"
                )

            cos = cos[position_ids]
            sin = sin[position_ids]

        return cos.to(x.dtype), sin.to(x.dtype)
