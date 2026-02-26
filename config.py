
# eeg_fm/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import Any, Dict, Optional, Tuple


@dataclass
class EEGModelConfig:
    # --------- Signal / tokenization ----------
    sample_rate: int = 200  # fixed in preprocessing
    patch_seconds: float = 1.0
    patch_hop_seconds: float = 1.0  # hop < patch_seconds => overlap
    max_tokens: int = 4096

    # --------- Model options ----------
    mlp_type: str = "swiglu"  # "gelu", "swiglu", or "geglu"
    norm_type: str = "rmsnorm"  # "layernorm" or "rmsnorm"
    layerscale_init: float = 0.0  # if >0, use LayerScale with this init value; else disable (typically starts with 1e-5; deeper models may need smaller)

    # --------- Model size ----------
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # --------- RoPE (time only) ----------
    rope_theta: float = 10000.0
    rotary_pct: float = 1.0

    # --------- Spatial embedding ----------
    coord_num_freqs: int = 6
    coord_max_freq: float = 2.0
    coord_include_raw: bool = True
    coord_jitter_std: float = 0.05           # applied with prob coord_jitter_prob
    coord_jitter_prob: float = 0.5
    coord_renormalize: bool = False           # renormalize coords to unit cube after jitter

    # --------- Frequency features ----------
    freq_min_hz: float = 0.5
    freq_max_hz: float = 50.0  # default; consider 45-50 for scalp EEG
    freq_bins: int = 32
    freq_spacing: str = "log"          # "log" or "linear"
    freq_use_scale: bool = True        # append mean(log-power) as scale dim
    freq_eps: float = 1e-6
    fft_dtype: str = "float32"         # float32 recommended for FFT stability

    # --------- FiLM fusion ----------
    film_hidden: int = 512

    # --------- JEPA predictor ----------
    predictor_hidden: int = 2048
    predictor_depth: int = 2  # for MLP predictor
    predictor_type: str = "cross_attn"  # "cross_attn" (I-JEPA) or "mlp"
    predictor_layers: int = 2            # cross-attn predictor depth
    predictor_n_heads: int = 8
    predictor_mlp_ratio: float = 4.0

    # --------- Mask token ----------
    mask_token_init_std: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "EEGModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return EEGModelConfig(**d)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class TrainConfig:
    # --------- Data ----------
    data_root: str = "/mnt/e/open_eeg"
    cache_dir: str = "/mnt/c/wds_cache"
    shard_glob: str = "**/*.tar"

    shard_shuffle: int = 500
    sample_shuffle: int = 256 # sample_shuffle 너무 크게 잡지 말기(예: 256~1024부터)
    cache_max_bytes: int = int(170*1024**3) # ex: 500GB -> 500*1024**3
    post_split_shuffle: int = 128
    eviction_interval: int = 8

    base_seconds: int = 10
    split_long_prob: float = 0.5

    enable_channel_grouping: bool = True

    # bucket batching
    tokens_per_batch: int = 16384           # target sum(valid_tokens) per microbatch
    max_samples_per_batch: int = 256
    # bucket_boundaries: str = "0,200,400,800,1200,2000,4096"
    # allow_token_overshoot_ratio: float = 1.10 # overshoot 방지용 greedy packing (tokens_per_batch를 크게 초과하지 않게)
    # padded_tokens_per_batch: int = 0 # optional padded-token budget (batch_size * max_len). 0 disables.

    # auto-tune tokens_per_batch with a synthetic probe (optional)
    auto_tune_tokens_per_batch: bool = False
    auto_tune_target_mem_frac: float = 0.85
    auto_tune_probe_seq_len: int = 4096  # worst-case per-sample tokens (<= model_cfg.max_tokens)
    auto_tune_probe_batch: int = 2

    num_workers: int = 8 # num_workers는 2GPU 기준 4~6부터 시작(8은 RAM 여유 있을 때)

    # --------- JEPA masking ----------
    mask_time_prob: float = 0.8
    mask_spatial_prob: float = 0.8
    mask_dilate_time: int = 0  # if patches overlap, set 1 to reduce leakage via neighbors
    time_mask_ratio_min: float = 0.15
    time_mask_ratio_max: float = 0.35
    spatial_mask_ratio_min: float = 0.10
    spatial_mask_ratio_max: float = 0.30

    # --------- Student augmentations (NEW A) ----------
    aug_gain_min: float = 0.8
    aug_gain_max: float = 1.2
    aug_channel_gain_std: float = 0 # 0.05   # per-channel multiplicative jitter std
    aug_noise_std_min: float = 0.00          # relative to RMS
    aug_noise_std_max: float = 0.03
    aug_channel_drop_prob: float = 0.05      # per-channel dropout prob

    # --------- Freq corruption (student only) ----------
    freq_domain_drop_prob: float = 0.5
    freq_physio_mask_prob: float = 0.6
    freq_num_bands_min: int = 1
    freq_num_bands_max: int = 2
    freq_random_width_min: float = 0.10
    freq_random_width_max: float = 0.25

    # --------- Optimization ----------
    lr: float = 2e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    grad_accum_steps: int = 1

    max_steps: int = 200_000
    warmup_steps: int = 5_000

    # EMA teacher
    ema_momentum: float = 0.996
    ema_momentum_final: float = 0.9999

    # --------- Runtime ----------
    output_dir: str = "./checkpoints/eeg_jepa"
    log_every: int = 50
    save_every: int = 5000

    mixed_precision: str = "bf16"
    use_wandb: bool = True
    wandb_project: str = "eeg-foundation"
    run_name: Optional[str] = None

    # Debug / small runs
    limit_num_samples: int = 0

    def bucket_bounds(self):
        return [int(x) for x in self.bucket_boundaries.split(",") if x.strip()]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return TrainConfig(**d)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
