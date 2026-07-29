import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply split-half rotation used by LLaMA-style RoPE."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head dimension, got {x.shape[-1]}"
        )

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE to query and key tensors.

    Args:
        q: Query tensor with shape (B, H, T, D).
        k: Key tensor with shape (B, H, T, D).
        cos: Cosine tensor with shape (B, T, D) or (T, D).
        sin: Sine tensor with shape (B, T, D) or (T, D).

    Returns:
        Rotated query and key tensors.
    """
    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same shape, got {q.shape} and {k.shape}"
        )

    if cos.ndim == 2:
        # (T, D) -> (1, 1, T, D)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif cos.ndim == 3:
        # (B, T, D) -> (B, 1, T, D)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    else:
        raise ValueError(
            "cos and sin must have shape (T, D) or (B, T, D), "
            f"got cos={cos.shape}, sin={sin.shape}"
        )

    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated
