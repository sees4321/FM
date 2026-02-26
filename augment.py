
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

    # channel dropout (k-drop)
    if channel_drop_prob > 0:
        # expected drops = C*prob를 참고해서 k를 제한
        k = max(0, min(C - 1, int(round(C * channel_drop_prob))))
        if k > 0:
            drop = torch.zeros((B, C), device=device, dtype=torch.bool)
            for b in range(B):
                idx = torch.randperm(C, device=device)[:k]
                drop[b, idx] = True
            x_fp32 = x_fp32.masked_fill(drop[:, :, None], 0.0)

    # gaussian noise (relative to RMS)
    if noise_std_max > 0:
        # RMS per sample (avoid pad influence? 여기서는 전체 T 사용)
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=(1, 2), keepdim=True) + 1e-8)  # (B,1,1)
        ns = torch.empty((B, 1, 1), device=device).uniform_(noise_std_min, noise_std_max)
        noise = torch.randn_like(x_fp32) * (rms * ns)
        x_fp32 = x_fp32 + noise

    return x_fp32.to(x.dtype)
