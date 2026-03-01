
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

    # --------- Encoder attention architecture ----------
    # "full"      : standard 1D packed-token self-attention (baseline)
    # "divided"   : TimeSformer-like divided attention (temporal pass then spatial pass)
    # "hybrid"    : mostly divided, but every N blocks use a full-attention block
    encoder_arch: str = "hybrid"
    full_attn_every: int = 4  # for hybrid; 0 disables full-attn blocks
    full_attn_use_spatial_bias: bool = True  # may disable flash SDP depending on PyTorch version

    # --------- Spatial attention bias ----------
    # Additive attention-logits bias computed from electrode coords.
    # - "legendre": learnable Legendre series in cos(gamma)=u·v (unit-sphere)
    # - "none"    : disable
    # NOTE: the previous optional "mlp" spatial bias was removed for efficiency.
    spatial_bias_type: str = "legendre"  # "legendre" | "none"
    spatial_bias_degree: int = 8
    spatial_bias_use_unit_sphere: bool = True
    spatial_bias_scale: float = 1.0
    # (deprecated, kept for backward compatibility)
    spatial_bias_mlp_hidden: int = 64
    spatial_bias_mlp_depth: int = 2

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
    # --------- Repro / determinism ----------
    # NOTE:
    # - Pretraining throughput is usually better with cudnn_benchmark=True and deterministic=False.
    # - For ablations/tuning, fixing seeds is recommended.
    seed: int = 42
    torch_deterministic: bool = False
    cudnn_benchmark: bool = True

    # --------- Data ----------
    data_root: str = "/mnt/e/open_eeg"
    cache_dir: str = "/mnt/c/wds_cache"
    shard_glob: str = "**/*.tar"

    shard_shuffle: int = 500
    sample_shuffle: int = 256 # sample_shuffle 너무 크게 잡지 말기(예: 256~1024부터)
    cache_max_bytes: int = int(170*1024**3) # ex: 500GB -> 500*1024**3
    post_split_shuffle: int = 128
    eviction_interval: int = 8

    # ------------------------------------------------------------
    # Long-segment (60s) handling
    # ------------------------------------------------------------
    # 기존 split-long은 oversampling을 만들기 쉬워서,
    # baseline에서는 60s 샘플을 확률적으로 random window crop(10/30s) 하도록 했었음.
    #
    # 이번 버전에서는 DINO-style multi-crop(=global + local crops)을 기본으로 추가.
    #
    # long_multicrop_mode:
    #   - "off"      : multi-crop 비활성 (아래 deprecated random crop 사용 가능)
    #   - "expand"   : crops를 독립 샘플로 확장(flatmap)만 함 (추천 시작점)
    #   - "multiview": expand + (옵션) multi-view consistency loss 추가
    base_seconds: int = 10

    # legacy split-long (oversampling 위험; 필요 시만 켜기)
    split_long_prob: float = 0.0  # deprecated (kept for backward compatibility)

    # legacy random window crop (deprecated): multi-crop이 off일 때만 적용
    long_crop_prob: float = 0.0   # only applied when duration_sec==60 and long_multicrop_mode=="off"
    long_crop_30_prob: float = 0.5

    # NEW: DINO-style multi-crop for 60s
    long_multicrop_mode: str = "expand"     # "off" | "expand" | "multiview"
    long_multicrop_prob: float = 0.3        # P(add local crops | duration==60)
    long_multicrop_n_global: int = 1
    long_multicrop_global_sec: int = 30
    long_multicrop_n_local: int = 4
    long_multicrop_local_sec: int = 10
    long_multicrop_local_within_global: bool = True  # local crops sampled inside the chosen global window

    # multi-view training (only when long_multicrop_mode == "multiview")
    multiview_loss_weight: float = 0.05
    multiview_only_for_multicrop: bool = True  # do not try to pair non-multicrop samples

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

    # NEW: token-budget based gradient accumulation (dynamic)
    # - Accumulate multiple micro-batches (possibly from different (C,P) buckets)
    # - Perform optimizer.step() when accumulated token-count reaches the budget.
    # Basis:
    #   "target"  : use #target tokens (recommended for JEPA since loss defined on targets)
    #   "valid"   : use #valid tokens (C*P)
    accum_tokens_basis: str = "target"
    tokens_per_update: int = 8192

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

    # --------- Logging ----------
    log_bucket_key: bool = True
    log_proxies: bool = True
    proxy_max_tokens: int = 2048  # subsample target tokens for cheap proxy metrics (0 disables subsampling)

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
