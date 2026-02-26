# eeg_fm/train.py
from __future__ import annotations

import argparse
import copy
import math
import os
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .config import EEGModelConfig, TrainConfig
from .model import EEGEncoder, CrossAttentionPredictor
from .masking import (
    sample_jepa_target_mask,
    sample_freq_bin_mask,
    dilate_time_mask,
)
from .augment import apply_student_augmentations
from .data import find_shards, build_webdataset, ShapeBatcher #, AdaptiveTokenBucketBatcher

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
except Exception:
    Accelerator = None


def set_torch_flags_for_sdp():
    # flash / mem-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def ema_momentum_schedule(step: int, total: int, m0: float, m1: float) -> float:
    progress = step / max(1, total)
    m = m1 - (m1 - m0) * (0.5 * (1.0 + math.cos(math.pi * progress)))
    return float(m)


@torch.no_grad()
def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, m: float):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))


@torch.no_grad()
def mask_to_packed_indices(mask: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    mask/valid: (B,C,P) bool

    returns:
      c_idx: (B,Lmax) long, channel index (>=0 for valid positions; 0 for PAD)
      t_idx: (B,Lmax) long, time-patch index (>=0; 0 for PAD)
      pad:   (B,Lmax) bool, True=PAD
    Ordering: sort by (t, c) for determinism.
    """
    device = mask.device
    mask = mask & valid
    B, C, P = mask.shape
    lengths = mask.flatten(1).sum(dim=1)  # (B,)
    Lmax = int(lengths.max().item())
    if Lmax <= 0:
        # should not happen if masks are sane; fall back to 1 dummy token
        Lmax = 1

    c_idx = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    t_idx = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    pad = torch.ones((B, Lmax), dtype=torch.bool, device=device)

    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False)  # (L,2): [c,t]
        if idx.numel() == 0:
            continue
        # sort by time then channel
        key = idx[:, 1] * C + idx[:, 0]
        idx = idx[key.argsort()]
        l = idx.shape[0]
        c_idx[b, :l] = idx[:, 0]
        t_idx[b, :l] = idx[:, 1]
        pad[b, :l] = False

    return c_idx, t_idx, pad


@torch.no_grad()
def rescale_small_segments(
    x: torch.Tensor,            # (B,C,T) fp16/bf16/fp32
    target_rms: float = 1.0,
    rms_low: float = 0.5,
    rms_floor: float = 0.05,
    gain_max: float = 8.0,
    clip: float = 10.0,
) -> torch.Tensor:
    # fp32에서 통계 계산(안정)
    x32 = x.float()
    # rms = torch.sqrt(torch.mean(x32 * x32, dim=(1,2), keepdim=True) + 1e-8)  # (B,1,1)
    rms = torch.sqrt(torch.mean(x32 * x32, dim=-1, keepdim=True) + 1e-8)  # (B,C,1)

    # rms가 너무 작은 것만 보정
    need = (rms < rms_low).to(x32.dtype)  # (B,1,1) 0/1
    gain = (target_rms / rms.clamp_min(rms_floor)).clamp(1.0 / gain_max, gain_max)

    # need==1인 샘플만 스케일 적용
    x32 = x32 * (1.0 + need * (gain - 1.0))

    if clip and clip > 0:
        x32 = x32.clamp(-clip, clip)

    return x32.to(dtype=x.dtype)


def gather_channel_embeddings(x: torch.Tensor, c_idx: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,D)
    c_idx: (B,L) long (>=0)
    pad: (B,L) bool
    returns: (B,L,D) with pad->0
    """
    B, C, D = x.shape
    B2, L = c_idx.shape
    assert B == B2
    idx = c_idx[..., None].expand(B, L, D)
    out = x.gather(dim=1, index=idx)
    return out.masked_fill(pad[..., None], 0.0)


def auto_tune_tokens_per_batch(
    student: EEGEncoder,
    predictor: CrossAttentionPredictor,
    model_cfg: EEGModelConfig,
    train_cfg: TrainConfig,
    accelerator: Accelerator,
) -> int:
    """
    NEW(1): Synthetic probe to estimate a safe tokens_per_batch for the *worst bucket* (seq_len ~= max_tokens).
    This is a heuristic. It tends to be conservative and avoids OOM surprises.

    Returns the (possibly updated) tokens_per_batch.
    """
    if not train_cfg.auto_tune_tokens_per_batch:
        return train_cfg.tokens_per_batch
    if accelerator.device.type != "cuda":
        return train_cfg.tokens_per_batch

    device = accelerator.device
    # worst-case per-sample tokens (bounded by model_cfg.max_tokens)
    L = int(min(train_cfg.auto_tune_probe_seq_len, model_cfg.max_tokens))
    L = max(256, L)
    Bp = int(max(1, train_cfg.auto_tune_probe_batch))

    # worst-case (memory) happens when context is largest -> target fraction is smallest.
    tgt_frac_min = float(min(train_cfg.time_mask_ratio_min, train_cfg.spatial_mask_ratio_min))
    ctx_frac = float(max(0.50, min(0.95, 1.0 - tgt_frac_min)))

    Lc = int(max(64, round(L * ctx_frac)))
    Lt = int(max(32, L - Lc))

    dtype = torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16

    # synthetic tensors (no patch embed / FFT) just to measure transformer activations
    ctx = torch.randn((Bp, Lc, model_cfg.d_model), device=device, dtype=dtype)
    ctx_pad = torch.zeros((Bp, Lc), device=device, dtype=torch.bool)
    rope_ctx = torch.arange(Lc, device=device, dtype=torch.long)[None, :].expand(Bp, Lc)

    tgt_coord = torch.randn((Bp, Lt, model_cfg.d_model), device=device, dtype=dtype)
    tgt_pad = torch.zeros((Bp, Lt), device=device, dtype=torch.bool)
    rope_tgt = torch.arange(Lt, device=device, dtype=torch.long)[None, :].expand(Bp, Lt)

    # reset peak stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    student.train()
    predictor.train()

    # forward/backward
    z = student(ctx, padding_mask=ctx_pad, rope_pos=rope_ctx)
    pred = predictor(z, ctx_pad, rope_ctx, tgt_coord, tgt_pad, rope_tgt)
    loss = (pred ** 2).mean()
    loss.backward()

    mem_used = torch.cuda.max_memory_allocated(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * float(train_cfg.auto_tune_target_mem_frac))

    # clear grads to avoid later surprises
    student.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    tokens_probe = Bp * L
    if mem_used <= 0:
        return train_cfg.tokens_per_batch

    scale = target_mem / float(mem_used)
    tokens_safe = int(tokens_probe * scale * 0.90)  # extra safety margin
    tokens_safe = max(1024, tokens_safe)

    # round down for stability
    round_to = 256
    tokens_safe = (tokens_safe // round_to) * round_to

    if accelerator.is_main_process:
        print(f"[auto_tune] probe: B={Bp}, L={L} (Lc={Lc}, Lt={Lt}) mem_used={mem_used/1e9:.2f}GB "
              f"target_mem={target_mem/1e9:.2f}GB -> tokens_per_batch~{tokens_safe}")

    # only reduce (avoid unexpected huge changes)
    if train_cfg.tokens_per_batch > tokens_safe:
        return tokens_safe
    return train_cfg.tokens_per_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_cfg", type=str, default="")
    p.add_argument("--train_cfg", type=str, default="")

    # overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tokens_per_batch", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if Accelerator is None:
        raise RuntimeError("accelerate is not installed. pip install accelerate")

    model_cfg = EEGModelConfig() if not args.model_cfg else EEGModelConfig.from_json(args.model_cfg)
    train_cfg = TrainConfig() if not args.train_cfg else TrainConfig.from_json(args.train_cfg)
    seed = getattr(train_cfg, "seed", 42)
    set_seed(seed, device_specific=True)

    # overrides
    if args.data_root is not None: train_cfg.data_root = args.data_root
    if args.cache_dir is not None: train_cfg.cache_dir = args.cache_dir
    if args.output_dir is not None: train_cfg.output_dir = args.output_dir

    if args.d_model is not None: model_cfg.d_model = args.d_model
    if args.n_layers is not None: model_cfg.n_layers = args.n_layers
    if args.n_heads is not None: model_cfg.n_heads = args.n_heads

    if args.lr is not None: train_cfg.lr = args.lr
    if args.tokens_per_batch is not None: train_cfg.tokens_per_batch = args.tokens_per_batch
    if args.grad_accum_steps is not None: train_cfg.grad_accum_steps = args.grad_accum_steps
    if args.max_steps is not None: train_cfg.max_steps = args.max_steps

    if args.use_wandb: train_cfg.use_wandb = True
    if args.no_wandb: train_cfg.use_wandb = False
    if args.run_name is not None: train_cfg.run_name = args.run_name

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.grad_accum_steps,
        mixed_precision=train_cfg.mixed_precision,
        log_with="wandb" if train_cfg.use_wandb else None,
        project_dir=train_cfg.output_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(train_cfg.output_dir, exist_ok=True)
        model_cfg.save_json(os.path.join(train_cfg.output_dir, "model_config.json"))
        train_cfg.save_json(os.path.join(train_cfg.output_dir, "train_config.json"))

    if train_cfg.use_wandb:
        accelerator.init_trackers(
            train_cfg.wandb_project,
            config={**model_cfg.to_dict(), **train_cfg.to_dict()},
            init_kwargs={"wandb": {"name": train_cfg.run_name}} if train_cfg.run_name else None,
        )

    set_torch_flags_for_sdp()

    # models (move to device before dataloader to allow auto-tune)
    student = EEGEncoder(model_cfg).to(accelerator.device)
    predictor = CrossAttentionPredictor(model_cfg).to(accelerator.device)

    # optional: auto-tune tokens_per_batch
    tuned = auto_tune_tokens_per_batch(student, predictor, model_cfg, train_cfg, accelerator)
    train_cfg.tokens_per_batch = tuned

    # EMA teacher (after potential tuning)
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # data
    shards = find_shards(train_cfg.data_root, train_cfg.shard_glob)
    if len(shards) == 0:
        raise RuntimeError(f"No shards found under {train_cfg.data_root} with {train_cfg.shard_glob}")

    patch_samples = int(round(model_cfg.sample_rate * model_cfg.patch_seconds))
    hop_sec = float(getattr(model_cfg, "patch_hop_seconds", model_cfg.patch_seconds))
    if hop_sec <= 0:
        hop_sec = float(model_cfg.patch_seconds)
    hop_samples = int(round(model_cfg.sample_rate * hop_sec))
    hop_samples = max(1, hop_samples)

    ds = build_webdataset(
        shards=shards,
        cache_dir=train_cfg.cache_dir,
        shard_shuffle=train_cfg.shard_shuffle,
        sample_shuffle=train_cfg.sample_shuffle,
        base_seconds=train_cfg.base_seconds,
        split_long_prob=train_cfg.split_long_prob,
        max_tokens=model_cfg.max_tokens,
        patch_samples=patch_samples,
        hop_samples=hop_samples,
        enable_channel_grouping=train_cfg.enable_channel_grouping,
        limit_num_samples=train_cfg.limit_num_samples,
        cache_max_bytes=train_cfg.cache_max_bytes,
        post_split_shuffle=train_cfg.post_split_shuffle,
        eviction_interval=train_cfg.eviction_interval,
    )

    # bucket batching
    ds_batched = ShapeBatcher(
        dataset=ds,
        tokens_per_batch=train_cfg.tokens_per_batch,
        max_samples_per_batch=train_cfg.max_samples_per_batch,
        patch_samples=patch_samples,
        hop_samples=hop_samples,

        # flush 정책(추천 시작값)
        max_wait_samples=5000,          # 희귀 shape가 5000샘플 동안 배치 못 만들면 방출
        flush_check_every=256,          # 256샘플마다 expired 검사
        max_pending_samples=512,        # CPU 메모리 보호(중요!)
        max_pending_tokens=0,           # 필요하면 활성화(예: 2_000_000)

        shuffle_within_bucket=True,
        yield_incomplete=True,
    )
    # bucket_bounds = train_cfg.bucket_bounds()
    # ds_batched = AdaptiveTokenBucketBatcher(
    #     dataset=ds,
    #     boundaries=bucket_bounds,
    #     tokens_per_batch=train_cfg.tokens_per_batch,
    #     max_samples_per_batch=train_cfg.max_samples_per_batch,
    #     patch_samples=patch_samples,
    #     hop_samples=hop_samples,
    #     allow_token_overshoot_ratio=train_cfg.allow_token_overshoot_ratio,
    #     padded_tokens_per_batch=train_cfg.padded_tokens_per_batch,
    # )

    import webdataset as wds
    loader = wds.WebLoader(ds_batched, batch_size=None, num_workers=train_cfg.num_workers)

    # optimizer
    opt = AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    # accelerate prepare
    student, teacher, predictor, opt, loader = accelerator.prepare(student, teacher, predictor, opt, loader)

    # freq bin centers for masking
    bin_centers = accelerator.unwrap_model(student).freq_feat.bin_centers_hz.detach().cpu()

    student.train()
    predictor.train()
    teacher.eval()

    pbar = tqdm(total=train_cfg.max_steps, disable=not accelerator.is_local_main_process)
    it = iter(loader)
    global_step = 0

    while global_step < train_cfg.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch["eeg"]          # (B,C,T)
        coords = batch["coord"]   # (B,C,3)
        x = rescale_small_segments(x, target_rms=1.0, rms_low=-0.5, rms_floor=0.05, gain_max=8.0, clip=15.0)
        n_channels = batch["n_channels"].to(x.device)
        n_patches = batch["n_patches"].to(x.device)
        B, C_max, _ = x.shape
        P_max = int(n_patches.max().item())

        # valid token mask (B,C,P)
        chan_ok = (torch.arange(C_max, device=x.device)[None, :, None] < n_channels[:, None, None])
        time_ok = (torch.arange(P_max, device=x.device)[None, None, :] < n_patches[:, None, None])
        valid = chan_ok & time_ok  # (B,C,P)

        with accelerator.accumulate(student):
            # 1) JEPA target mask (B,C,P)
            target_mask = sample_jepa_target_mask(
                coords=coords.to(x.device),
                n_channels=n_channels,
                n_patches=n_patches,
                mask_time_prob=train_cfg.mask_time_prob,
                mask_spatial_prob=train_cfg.mask_spatial_prob,
                time_ratio_range=(train_cfg.time_mask_ratio_min, train_cfg.time_mask_ratio_max),
                spatial_ratio_range=(train_cfg.spatial_mask_ratio_min, train_cfg.spatial_mask_ratio_max),
            )
            # optional dilation to reduce overlap leakage
            if train_cfg.mask_dilate_time and train_cfg.mask_dilate_time > 0:
                target_mask = dilate_time_mask(target_mask, dilation=int(train_cfg.mask_dilate_time))

            target_mask = target_mask & valid
            context_mask = (~target_mask) & valid

            # 2) student augmentations (time alignment preserving)
            x_aug = apply_student_augmentations(
                x,
                gain_min=train_cfg.aug_gain_min,
                gain_max=train_cfg.aug_gain_max,
                channel_gain_std=train_cfg.aug_channel_gain_std,
                noise_std_min=train_cfg.aug_noise_std_min,
                noise_std_max=train_cfg.aug_noise_std_max,
                channel_drop_prob=train_cfg.aug_channel_drop_prob,
                polarity_flip_prob=train_cfg.aug_polarity_flip_prob,
            )

            # 3) freq corruption (student context only)
            freq_domain_drop = (torch.rand((B,), device=x.device) < train_cfg.freq_domain_drop_prob)
            freq_mask_bins = sample_freq_bin_mask(
                B=B,
                K=model_cfg.freq_bins,
                bin_centers_hz=bin_centers,
                physio_prob=train_cfg.freq_physio_mask_prob,
                num_bands_min=train_cfg.freq_num_bands_min,
                num_bands_max=train_cfg.freq_num_bands_max,
                random_width_min=train_cfg.freq_random_width_min,
                random_width_max=train_cfg.freq_random_width_max,
                device=x.device,
            )

            # 4) pack indices
            c_ctx, t_ctx, pad_ctx = mask_to_packed_indices(context_mask, valid)
            c_tgt, t_tgt, pad_tgt = mask_to_packed_indices(target_mask, valid)

            with accelerator.autocast():
                # 5) student: embed+encode context only
                tok_ctx, pad_ctx, rope_ctx = accelerator.unwrap_model(student).embed_from_indices(
                    x=x_aug,
                    coords=coords.to(x.device),
                    c_idx=c_ctx,
                    t_idx=t_ctx,
                    pad=pad_ctx,
                    freq_mask_bins=freq_mask_bins,
                    freq_domain_drop=freq_domain_drop,
                )
                # NOTE: embed_from_indices called on unwrap_model(student) to avoid DDP wrapper issues with .unfold()
                # Then we pass tokens through the wrapped model for proper distributed behavior.
                z_ctx = student(tok_ctx, padding_mask=pad_ctx, rope_pos=rope_ctx)  # (B,Lc,D)

                # 6) teacher: embed+encode targets only (no corruption)
                with torch.no_grad():
                    tok_tgt, pad_tgt2, rope_tgt = accelerator.unwrap_model(teacher).embed_from_indices(
                        x=x,
                        coords=coords.to(x.device),
                        c_idx=c_tgt,
                        t_idx=t_tgt,
                        pad=pad_tgt,
                        freq_mask_bins=None,
                        freq_domain_drop=None,
                    )
                    z_tgt = teacher(tok_tgt, padding_mask=pad_tgt2, rope_pos=rope_tgt)  # (B,Lt,D)

                # 7) predictor: build target queries from coord embeddings + cross-attend to ctx
                coord_ch = accelerator.unwrap_model(student).coord_embed(coords.to(x.device))  # (B,C,D)
                coord_tgt = gather_channel_embeddings(coord_ch, c_tgt.clamp(min=0), pad_tgt)   # (B,Lt,D)

                pred_tgt = predictor(
                    ctx=z_ctx,
                    ctx_pad=pad_ctx,
                    rope_ctx=rope_ctx,
                    tgt_coord_emb=coord_tgt,
                    tgt_pad=pad_tgt,
                    rope_tgt=rope_tgt,
                )  # (B,Lt,D)

                # 8) loss on non-pad targets
                valid_tgt = ~pad_tgt
                loss = F.l1_loss(pred_tgt[valid_tgt], z_tgt[valid_tgt], reduction="mean")

            accelerator.backward(loss) # end autocast

            if accelerator.sync_gradients:
                if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                    accelerator.clip_grad_norm_(list(student.parameters()) + list(predictor.parameters()), train_cfg.grad_clip)

                lr = cosine_warmup(global_step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr

                opt.step()
                opt.zero_grad(set_to_none=True)

                m = ema_momentum_schedule(global_step, train_cfg.max_steps, train_cfg.ema_momentum, train_cfg.ema_momentum_final)
                update_ema(teacher=accelerator.unwrap_model(teacher), student=accelerator.unwrap_model(student), m=m)

                if accelerator.is_main_process and (global_step % train_cfg.log_every == 0):
                    logs = {
                        "loss": float(loss.detach().cpu().item()),
                        "lr": lr,
                        "ema_m": m,
                        "ctx_tokens_max": int((~pad_ctx).sum(dim=1).max().detach().cpu().item()),
                        "tgt_tokens_max": int((~pad_tgt).sum(dim=1).max().detach().cpu().item()),
                        "ctx_tokens_sum": int((~pad_ctx).sum().detach().cpu().item()),
                        "tgt_tokens_sum": int((~pad_tgt).sum().detach().cpu().item()),
                        "batch_size_samples": int(B),
                        "tokens_per_batch_cfg": int(train_cfg.tokens_per_batch),
                        "patch_samples": int(patch_samples),
                        "hop_samples": int(hop_samples),
                    }
                    accelerator.log(logs, step=global_step)

                if accelerator.is_main_process and (global_step % train_cfg.save_every == 0):
                    accelerator.wait_for_everyone()
                    ckpt_dir = os.path.join(train_cfg.output_dir, f"step_{global_step:07d}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.unwrap_model(student).save_pretrained(os.path.join(ckpt_dir, f"student_{global_step:07d}"))
                    torch.save(accelerator.unwrap_model(predictor).state_dict(), os.path.join(ckpt_dir, f"predictor_{global_step:07d}.pt"))
                    accelerator.unwrap_model(teacher).save_pretrained(os.path.join(ckpt_dir, f"teacher_{global_step:07d}"))
                    accelerator.save_state(os.path.join(ckpt_dir, f"accelerator_state_{global_step:07d}"))

                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}", "lr": f"{lr:.2e}"})

    pbar.close()

    if accelerator.is_main_process:
        accelerator.wait_for_everyone()  # ensure all processes have finished saving before we write the final checkpoint
        final_dir = os.path.join(train_cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(student).save_pretrained(os.path.join(final_dir, "student"))
        torch.save(accelerator.unwrap_model(predictor).state_dict(), os.path.join(final_dir, "predictor.pt"))
        accelerator.unwrap_model(teacher).save_pretrained(os.path.join(final_dir, "teacher"))
        accelerator.save_state(os.path.join(final_dir, "accelerator_state"))

if __name__ == "__main__":
    main()
