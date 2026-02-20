
# eeg_fm/augment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@torch.no_grad()
def apply_student_augmentations(
    x: torch.Tensor,  # (B,C,T) float16/float32
    gain_min: float,
    gain_max: float,
    channel_gain_std: float,
    noise_std_min: float,
    noise_std_max: float,
    channel_drop_prob: float,
    polarity_flip_prob: float = 0.0,
) -> torch.Tensor:
    """
    NEW(A): student 쪽에만 적용하는 "시간 정렬을 깨지 않는" 안전한 augmentation.
    - sample-wise gain
    - per-channel gain jitter
    - gaussian noise (RMS-relative)
    - channel dropout
    - optional polarity flip

    NOTE:
    - time shift/warp 같은 augmentation은 JEPA target mask와의 상호작용(누설/경계 이동)을 만들 수 있어
      여기서는 의도적으로 제외.
    """
    if x.numel() == 0:
        return x

    device = x.device
    B, C, T = x.shape
    x_fp32 = x.to(torch.float32)

    # polarity flip
    if polarity_flip_prob > 0:
        flip = (torch.rand((B, 1, 1), device=device) < polarity_flip_prob).to(torch.float32)
        sign = 1.0 - 2.0 * flip  # 1 or -1
        x_fp32 = x_fp32 * sign

    # sample-wise gain
    if (gain_min != 1.0) or (gain_max != 1.0):
        g = torch.empty((B, 1, 1), device=device).uniform_(gain_min, gain_max)
        x_fp32 = x_fp32 * g

    # per-channel gain jitter
    if channel_gain_std > 0:
        cg = torch.randn((B, C, 1), device=device) * channel_gain_std + 1.0
        # clamp to avoid extreme
        cg = torch.clamp(cg, 0.5, 2.0)
        x_fp32 = x_fp32 * cg

    # channel dropout
    if channel_drop_prob > 0:
        drop = (torch.rand((B, C), device=device) < channel_drop_prob)  # True drop
        # ensure not all channels dropped
        all_drop = drop.all(dim=1)
        if all_drop.any():
            # keep one random channel
            idx = torch.randint(0, C, (int(all_drop.sum().item()),), device=device)
            drop[all_drop, :] = True
            drop[all_drop, idx] = False
        x_fp32 = x_fp32.masked_fill(drop[:, :, None], 0.0)

    # gaussian noise (relative to RMS)
    if noise_std_max > 0:
        # RMS per sample (avoid pad influence? 여기서는 전체 T 사용)
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=(1, 2), keepdim=True) + 1e-8)  # (B,1,1)
        ns = torch.empty((B, 1, 1), device=device).uniform_(noise_std_min, noise_std_max)
        noise = torch.randn_like(x_fp32) * (rms * ns)
        x_fp32 = x_fp32 + noise

    return x_fp32.to(x.dtype)
