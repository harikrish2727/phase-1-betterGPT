"""
Multi-head causal self-attention with Rotary Position Embeddings (RoPE),
implemented using PyTorch scaled_dot_product_attention (SDPA).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from src.utils.rope_helper import apply_rotary_pos_emb


class MHAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional padding.

    Mask convention:
        attention_mask[b, t] = True  -> real token
        attention_mask[b, t] = False -> padding token

    Notes:
        - With no padding mask, SDPA handles causality internally.
        - With padding, a combined causal-and-padding mask is created.
        - RoPE is applied to queries and keys, not values.
    """

    def __init__(
        self,
        emb_dim: int,
        head_count: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        out_bias: bool = False,
    ) -> None:
        super().__init__()

        if emb_dim <= 0:
            raise ValueError(
                f"emb_dim must be positive, got {emb_dim}"
            )

        if head_count <= 0:
            raise ValueError(
                f"head_count must be positive, got {head_count}"
            )

        if emb_dim % head_count != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by "
                f"head_count ({head_count})"
            )

        inferred_head_dim = emb_dim // head_count

        if head_dim is None:
            head_dim = inferred_head_dim
        elif head_dim != inferred_head_dim:
            raise ValueError(
                "head_dim must equal emb_dim // head_count. "
                f"Expected {inferred_head_dim}, got {head_dim}"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )

        self.emb_dim = emb_dim
        self.head_count = head_count
        self.head_dim = head_dim
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(
            emb_dim,
            3 * emb_dim,
            bias=qkv_bias,
        )

        self.out_proj = nn.Linear(
            emb_dim,
            emb_dim,
            bias=out_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x:
                Input tensor with shape (B, T, emb_dim).

            sin:
                RoPE sine values. Its exact shape depends on
                apply_rotary_pos_emb.

            cos:
                RoPE cosine values. Its exact shape depends on
                apply_rotary_pos_emb.

            attention_mask:
                Optional tensor with shape (B, T).

                True or 1:
                    Real token.

                False or 0:
                    Padding token.

        Returns:
            Tensor with shape (B, T, emb_dim).
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch, sequence, embedding), "
                f"got {tuple(x.shape)}"
            )

        batch_size, seq_len, input_dim = x.shape

        if input_dim != self.emb_dim:
            raise ValueError(
                f"Expected x.shape[-1] == {self.emb_dim}, "
                f"got {input_dim}"
            )

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)

        k = k.reshape(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)

        v = v.reshape(
            batch_size,
            seq_len,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        dropout_p = self.dropout_p if self.training else 0.0

        #no padding path
        if attention_mask is None:
            y = scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )

        #Padding path
        else:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape (B, T), "
                    f"got {tuple(attention_mask.shape)}"
                )

            expected_shape = (batch_size, seq_len)

            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    "attention_mask must have shape "
                    f"{expected_shape}, got "
                    f"{tuple(attention_mask.shape)}"
                )

            # True means real token.
            valid_tokens = attention_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

            # Padding-key mask
            # (B, T) -> (B, 1, 1, T)
    
            valid_keys = valid_tokens[:, None, None, :]
            # Causal mask
            # True means the query-key pair is allowed.

            # Shape: (1, 1, T, T)
            positions = torch.arange(
                seq_len,
                device=x.device,
            )

            causal_mask = (
                positions[None, :] <= positions[:, None]
            )[None, None, :, :]

            # Broadcasting produces shape (B, 1, T, T).
            allowed_mask = causal_mask & valid_keys

            # Safely handle padded query rows
            # With left padding, a padded query can have no valid
            # causal key. Give padded queries temporary access to
            # their own diagonal position, then zero their outputs.
            padded_queries = ~valid_tokens[:, None, :, None]

            diagonal = torch.eye(
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )[None, None, :, :]

            allowed_mask = torch.where(
                padded_queries,
                diagonal,
                allowed_mask,
            )

            # Causality already included in allowed_mask,
            # so is_causal must be False here.
            y = scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=allowed_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        # Combine attention heads
        
        # (B, heads, T, head_dim)
        #  (B, T, heads, head_dim)
        # (B, T, emb_dim)

        y = y.transpose(1, 2).contiguous()
        y = y.reshape(batch_size, seq_len, self.emb_dim)

        y = self.out_proj(y)

        if attention_mask is not None:
            y = y * valid_tokens.unsqueeze(-1).to(y.dtype)

        return y
