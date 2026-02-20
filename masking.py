
# eeg_fm/masking.py
from __future__ import annotations

from typing import Dict, Tuple

import torch


def _randint(low: int, high: int, device=None) -> int:
    return int(torch.randint(low, high, (1,), device=device).item())


@torch.no_grad()
def sample_time_block_mask(
    B: int,
    C: int,
    P_t: int,
    ratio_min: float,
    ratio_max: float,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((B, C, P_t), dtype=torch.bool, device=device)
    for b in range(B):
        frac = float(torch.empty((), device=device).uniform_(ratio_min, ratio_max).item())
        L = max(1, int(round(frac * P_t)))
        L = min(L, P_t)
        s = _randint(0, P_t - L + 1, device=device)
        mask[b, :, s : s + L] = True
    return mask


@torch.no_grad()
def sample_spatial_block_mask(
    coords: torch.Tensor,      # (B, C, 3)
    valid_chan: torch.Tensor,  # (B, C) bool
    P_t: int,
    ratio_min: float,
    ratio_max: float,
    device: torch.device,
) -> torch.Tensor:
    B, C, _ = coords.shape
    mask = torch.zeros((B, C, P_t), dtype=torch.bool, device=device)

    for b in range(B):
        valid_idx = torch.nonzero(valid_chan[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        Cb = int(valid_idx.numel())
        frac = float(torch.empty((), device=device).uniform_(ratio_min, ratio_max).item())
        k = max(1, int(round(frac * Cb)))
        k = min(k, Cb)

        center_i = valid_idx[_randint(0, Cb, device=device)]
        center = coords[b, center_i]

        d = torch.sum((coords[b] - center[None, :]) ** 2, dim=-1)
        d = d.masked_fill(~valid_chan[b], float("inf"))

        nn = torch.topk(d, k, largest=False).indices
        mask[b, nn, :] = True

    return mask


@torch.no_grad()
def sample_jepa_target_mask(
    coords: torch.Tensor,         # (B, C, 3)
    n_channels: torch.Tensor,     # (B,) int
    n_patches: torch.Tensor,      # (B,) int (P_t)
    mask_time_prob: float,
    mask_spatial_prob: float,
    time_ratio_range: Tuple[float, float],
    spatial_ratio_range: Tuple[float, float],
) -> torch.Tensor:
    device = coords.device
    B, C_max, _ = coords.shape
    P_t_max = int(n_patches.max().item())

    chan_ids = torch.arange(C_max, device=device)[None, :]
    valid_chan = chan_ids < n_channels[:, None]
    time_ids = torch.arange(P_t_max, device=device)[None, :]
    valid_time = time_ids < n_patches[:, None]
    valid_tok = valid_chan[:, :, None] & valid_time[:, None, :]

    target = torch.zeros((B, C_max, P_t_max), dtype=torch.bool, device=device)

    use_time = torch.rand((B,), device=device) < mask_time_prob
    use_spat = torch.rand((B,), device=device) < mask_spatial_prob
    none = ~(use_time | use_spat)
    if none.any():
        use_time = use_time | none

    if use_time.any():
        tmask = sample_time_block_mask(
            B=B, C=C_max, P_t=P_t_max,
            ratio_min=time_ratio_range[0], ratio_max=time_ratio_range[1],
            device=device,
        )
        target = target | (tmask & use_time[:, None, None])

    if use_spat.any():
        smask = sample_spatial_block_mask(
            coords=coords,
            valid_chan=valid_chan,
            P_t=P_t_max,
            ratio_min=spatial_ratio_range[0],
            ratio_max=spatial_ratio_range[1],
            device=device,
        )
        target = target | (smask & use_spat[:, None, None])

    target = target & valid_tok
    return target


@torch.no_grad()
def apply_patch_signal_mask(
    x: torch.Tensor,           # (B, C, T)
    target_mask: torch.Tensor, # (B, C, P_t)
    patch_samples: int,
) -> torch.Tensor:
    B, C, T = x.shape
    P_t = target_mask.shape[-1]
    T_need = P_t * patch_samples
    if T < T_need:
        pad = T_need - T
        x = torch.nn.functional.pad(x, (0, pad))
        T = x.shape[-1]
    x = x[:, :, :T_need]
    xp = x.view(B, C, P_t, patch_samples).clone()
    xp[target_mask] = 0.0
    return xp.view(B, C, T_need)


@torch.no_grad()
def physio_band_bin_masks(bin_centers_hz: torch.Tensor) -> Dict[str, torch.Tensor]:
    def m(lo, hi):
        return (bin_centers_hz >= lo) & (bin_centers_hz < hi)

    return {
        "delta": m(0.5, 4.0),
        "theta": m(4.0, 8.0),
        "alpha": m(8.0, 12.0),
        "beta":  m(13.0, 30.0),
        "gamma": m(30.0, 45.0),
    }


@torch.no_grad()
def sample_freq_bin_mask(
    B: int,
    K: int,
    bin_centers_hz: torch.Tensor,  # (K,)
    physio_prob: float,
    num_bands_min: int,
    num_bands_max: int,
    random_width_min: float,
    random_width_max: float,
    device: torch.device,
) -> torch.Tensor:
    masks = torch.zeros((B, K), dtype=torch.bool, device=device)
    band_masks = physio_band_bin_masks(bin_centers_hz.to(device))
    band_names = list(band_masks.keys())

    for b in range(B):
        if float(torch.rand((), device=device).item()) < physio_prob:
            nb = _randint(num_bands_min, num_bands_max + 1, device=device)
            chosen = torch.randperm(len(band_names), device=device)[:nb].tolist()
            mb = torch.zeros((K,), dtype=torch.bool, device=device)
            for idx in chosen:
                mb = mb | band_masks[band_names[idx]]
            masks[b] = mb
        else:
            wmin = max(1, int(round(random_width_min * K)))
            wmax = max(wmin, int(round(random_width_max * K)))
            w = _randint(wmin, wmax + 1, device=device)
            s = _randint(0, K - w + 1, device=device)
            masks[b, s : s + w] = True

    return masks


@torch.no_grad()
def dilate_time_mask(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    """Dilate a (B,C,P) mask along the time-patch axis by +/- dilation."""
    if dilation <= 0:
        return mask
    B, C, P = mask.shape
    out = mask.clone()
    for d in range(1, dilation + 1):
        out[..., d:] |= mask[..., :-d]
        out[..., :-d] |= mask[..., d:]
    return out
