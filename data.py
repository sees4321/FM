
# eeg_fm/data.py
from __future__ import annotations

import glob
import json
import os
import random
import shutil
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch

try:
    import webdataset as wds
except Exception:
    wds = None


EEG_KEY = "eeg.npy"
COORD_KEY = "coord.npy"
META_KEY = "meta.json"


def find_shards(data_root: str, shard_glob: str) -> List[str]:
    pat = os.path.join(data_root, shard_glob)
    shards = sorted(glob.glob(pat, recursive=True))
    return shards


def cache_shard_to_ssd(shard_path: str, cache_dir: str) -> str:
    """
    단순 캐시: shard를 SSD cache_dir로 복사 후 그 경로를 반환.
    - NEW(C)와는 별개: 여기 캐시는 'file이 없으면 복사' 수준.
    - 실제 SSD 용량 관리(LRU 등)는 이후 확장 권장.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(shard_path)
    cached = os.path.join(cache_dir, base)
    if not os.path.exists(cached):
        tmp = cached + ".tmp"
        shutil.copyfile(shard_path, tmp)
        os.replace(tmp, cached)
    return cached


def _as_torch(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise TypeError(type(x))


def decode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    eeg = sample[EEG_KEY]
    coord = sample[COORD_KEY]
    meta = sample.get(META_KEY, {})
    if isinstance(meta, (bytes, bytearray)):
        meta = json.loads(meta.decode("utf-8"))

    eeg = _as_torch(eeg).to(torch.float16)     # stored float16
    coord = _as_torch(coord).to(torch.float32)
    coord = coord - coord.mean(dim=0, keepdim=True)
    coord = coord / (coord.norm(dim=-1).mean().clamp_min(1e-6))

    return {"eeg": eeg, "coord": coord, "meta": meta}




def compute_num_patches(T: int, patch_samples: int, hop_samples: int) -> int:
    """Number of sliding-window patches.
    P = floor((T - patch)/hop) + 1. Requires T >= patch.
    """
    if T < patch_samples:
        return 0
    return int((T - patch_samples) // hop_samples + 1)


def maybe_split_long_to_base(
    ex: Dict[str, Any],
    base_seconds: int,
    split_prob: float,
) -> List[Dict[str, Any]]:
    eeg: torch.Tensor = ex["eeg"]  # (C,T)
    coord: torch.Tensor = ex["coord"]
    meta: Dict[str, Any] = ex.get("meta", {})

    fs = int(meta.get("fs", 200))
    duration_sec = int(eeg.shape[-1] / fs)

    # duration 정보가 없으면 split 안 함
    if duration_sec not in (base_seconds, 10, 30, 60):
        return [ex]

    if duration_sec <= base_seconds:
        return [ex]

    if random.random() >= split_prob:
        return [ex]

    chunk_samples = base_seconds * fs
    C, T = eeg.shape
    n_chunks = duration_sec // base_seconds
    outs = []
    for i in range(n_chunks):
        s = i * chunk_samples
        e = (i + 1) * chunk_samples
        if e > T:
            break
        out = {
            "eeg": eeg[:, s:e].contiguous(),
            "coord": coord,
            "meta": dict(meta),
        }
        out["meta"]["duration_sec"] = base_seconds
        out["meta"]["chunk_index"] = i
        out["meta"]["parent_duration_sec"] = duration_sec
        outs.append(out)
    return outs if len(outs) > 0 else [ex]


@torch.no_grad()
def farthest_point_grouping(coords: torch.Tensor, K: int) -> torch.Tensor:
    C = coords.shape[0]
    dmat = torch.cdist(coords, coords, p=2)
    chosen = torch.zeros((K,), dtype=torch.long)
    chosen[0] = 0
    min_dist = dmat[0].clone()
    for i in range(1, K):
        idx = torch.argmax(min_dist).item()
        chosen[i] = idx
        min_dist = torch.minimum(min_dist, dmat[idx])
    return chosen


def maybe_group_channels(
    ex: Dict[str, Any],
    max_tokens: int,
    patch_samples: int,
    hop_samples: int,
    prefer_durations_sec: tuple = (10, 30, 60),   # 너의 정책 반영
) -> Dict[str, Any]:
    """
    기존: max_tokens 초과 시 채널을 K로 줄임
    변경: 채널은 유지하고, 필요 시 시간 길이를 crop 해서 P_t를 줄임 (예: 60s -> 30s)
    """
    eeg: torch.Tensor = ex["eeg"]      # (C,T)
    coord: torch.Tensor = ex["coord"]  # (C,3)
    meta: Dict[str, Any] = ex.get("meta", {})

    C, T = eeg.shape
    P_t = compute_num_patches(T, patch_samples=patch_samples, hop_samples=hop_samples)
    if P_t <= 0:
        ex["n_tokens"] = 0
        ex["n_patches"] = 0
        ex["n_channels"] = int(C)
        return ex

    N = C * P_t
    if N <= max_tokens:
        ex["n_tokens"] = int(N)
        ex["n_patches"] = int(P_t)
        ex["n_channels"] = int(C)
        return ex

    # ---- 토큰 초과: 채널 유지, P를 줄이기 ----
    P_max = max_tokens // C  # 허용 가능한 최대 패치 수
    if P_max < 1:
        # 이 케이스는 C가 너무 커서 1패치도 못 넣는 상황.
        # 너의 데이터(C<=134)에서는 사실상 안 생김.
        ex["n_tokens"] = 0
        ex["n_patches"] = 0
        ex["n_channels"] = int(C)
        return ex

    # prefer durations 중 "가능한 가장 긴 길이" 선택 (예: 60->30)
    fs = int(meta.get("fs", 200))  # 전처리에서 200Hz 고정이라면 200으로 fallback OK
    best_P = None
    best_Tneed = None

    for sec in prefer_durations_sec:
        T_sec = sec * fs
        P_sec = compute_num_patches(T_sec, patch_samples=patch_samples, hop_samples=hop_samples)
        if P_sec <= P_max and P_sec <= P_t:
            if best_P is None or P_sec > best_P:
                best_P = P_sec
                best_Tneed = (P_sec - 1) * hop_samples + patch_samples

    if best_P is None:
        # prefer durations로 안 되면, P_max에 맞춰 crop (일반적 fallback)
        P_new = min(P_t, P_max)
        T_need = (P_new - 1) * hop_samples + patch_samples
    else:
        P_new = best_P
        T_need = best_Tneed

    if T_need <= 0:
        ex["n_tokens"] = 0
        ex["n_patches"] = 0
        ex["n_channels"] = int(C)
        return ex

    # 랜덤 crop (다양성 확보)
    if T > T_need:
        start = random.randint(0, T - T_need)
    else:
        start = 0

    eeg = eeg[:, start:start + T_need].contiguous()

    ex["eeg"] = eeg
    ex["coord"] = coord
    # meta 업데이트(디버깅/추적용)
    ex["meta"] = dict(meta)
    ex["meta"]["cropped_to_fit_tokens"] = True
    ex["meta"]["crop_start"] = int(start)

    # 재계산
    P_t2 = compute_num_patches(eeg.shape[1], patch_samples=patch_samples, hop_samples=hop_samples)
    ex["n_tokens"] = int(C * P_t2)
    ex["n_patches"] = int(P_t2)
    ex["n_channels"] = int(C)
    return ex



def collate_pad(batch: List[Dict[str, Any]], patch_samples: int, hop_samples: int) -> Dict[str, Any]:
    B = len(batch)
    n_channels = torch.tensor([b["n_channels"] for b in batch], dtype=torch.long)
    n_patches = torch.tensor([b["n_patches"] for b in batch], dtype=torch.long)

    C_max = int(n_channels.max().item())
    P_max = int(n_patches.max().item())
    T_max = (P_max - 1) * hop_samples + patch_samples if P_max > 0 else 0

    eeg_pad = torch.zeros((B, C_max, T_max), dtype=torch.float16)
    coord_pad = torch.zeros((B, C_max, 3), dtype=torch.float32)

    for i, b in enumerate(batch):
        eeg = b["eeg"]    # (C,T)
        coord = b["coord"]
        C, T = eeg.shape
        P = compute_num_patches(T, patch_samples=patch_samples, hop_samples=hop_samples)
        T_need = (P - 1) * hop_samples + patch_samples if P > 0 else 0
        eeg = eeg[:, :T_need]
        eeg_pad[i, :C, :T_need] = eeg
        coord_pad[i, :C, :] = coord

    return {
        "eeg": eeg_pad,
        "coord": coord_pad,
        "n_channels": n_channels,
        "n_patches": n_patches,
    }

def collate_stack(batch, patch_samples, hop_samples):
    # 여기서는 모든 샘플이 동일 (C,P)라고 가정
    B = len(batch)
    C = batch[0]["n_channels"]
    P = batch[0]["n_patches"]
    T_need = (P-1)*hop_samples + patch_samples

    eeg = torch.stack([b["eeg"][:, :T_need] for b in batch], dim=0)    # (B,C,T)
    coord = torch.stack([b["coord"] for b in batch], dim=0)           # (B,C,3)
    n_channels = torch.full((B,), C, dtype=torch.long)
    n_patches = torch.full((B,), P, dtype=torch.long)
    return {"eeg": eeg, "coord": coord, "n_channels": n_channels, "n_patches": n_patches}

class ShapeBatcher:
    def __init__(
        self,
        dataset: Iterable[Dict[str, Any]],
        tokens_per_batch: int,
        max_samples_per_batch: int,
        patch_samples: int,
        hop_samples: int,
    ):
        self.dataset = dataset
        self.tokens_per_batch = int(tokens_per_batch)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.hop_samples = int(hop_samples)
        self.patch_samples = int(patch_samples)

    def __iter__(self):
        buckets = dict()  # key -> list
        for ex in self.dataset:
            C = ex["n_channels"]; P = ex["n_patches"]
            key = (C, P)
            buckets.setdefault(key, []).append(ex)

            tokens_per_sample = C * P
            bs = max(1, min(self.max_samples_per_batch, self.tokens_per_batch // tokens_per_sample))

            if len(buckets[key]) >= bs:
                batch = buckets[key][:bs]
                buckets[key] = buckets[key][bs:]
                yield collate_stack(batch)  # padding 없이 stack

class AdaptiveTokenBucketBatcher:
    """
    NEW(C): token-length bucket batching + greedy packing
    - bucket 내에서 sum(valid_tokens)가 tokens_per_batch를 크게 초과하지 않게
      '추가하면 너무 초과되는 샘플'은 다음 배치로 넘김.
    - optional로 padded budget (batch_size * max_len)도 제한 가능.

    목적:
    - padding 최소화(버킷)
    - step당 compute 변동 감소(그리디/예산)
    """
    def __init__(
        self,
        dataset: Iterable[Dict[str, Any]],
        boundaries: List[int],
        tokens_per_batch: int,
        max_samples_per_batch: int,
        patch_samples: int,
        hop_samples: int,
        allow_token_overshoot_ratio: float = 1.10,
        padded_tokens_per_batch: int = 0,
    ):
        self.dataset = dataset
        self.boundaries = boundaries
        self.tokens_per_batch = int(tokens_per_batch)
        self.max_samples_per_batch = int(max_samples_per_batch)
        self.hop_samples = int(hop_samples)
        self.patch_samples = int(patch_samples)
        self.allow_token_overshoot_ratio = float(allow_token_overshoot_ratio)
        self.padded_tokens_per_batch = int(padded_tokens_per_batch)

        assert len(boundaries) >= 2
        assert self.tokens_per_batch > 0

    def _bucket_id(self, n_tokens: int) -> int:
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= n_tokens < self.boundaries[i + 1]:
                return i
        return len(self.boundaries) - 2

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        nb = len(self.boundaries) - 1
        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(nb)]
        sums = [0 for _ in range(nb)]
        maxlens = [0 for _ in range(nb)]  # max n_tokens in bucket

        for ex in self.dataset:
            n = int(ex.get("n_tokens", 0))
            if n <= 0:
                continue

            b = self._bucket_id(n)
            cur = buckets[b]
            cur_sum = sums[b]
            cur_max = maxlens[b]

            # NEW(C): if adding this sample would overshoot too much, flush current bucket first
            if len(cur) > 0:
                new_sum = cur_sum + n
                over_limit = new_sum > int(self.tokens_per_batch * self.allow_token_overshoot_ratio)

                # optional padded budget: (batch_size+1) * max(maxlen, n)
                if self.padded_tokens_per_batch > 0:
                    new_max = max(cur_max, n)
                    padded_est = (len(cur) + 1) * new_max
                    over_limit = over_limit or (padded_est > self.padded_tokens_per_batch)

                if over_limit:
                    batch = cur
                    buckets[b] = []
                    sums[b] = 0
                    maxlens[b] = 0
                    yield collate_pad(batch, patch_samples=self.patch_samples, hop_samples=self.hop_samples)

                    # reset after flush
                    cur = buckets[b]
                    cur_sum = sums[b]
                    cur_max = maxlens[b]

            # add sample
            cur.append(ex)
            sums[b] = cur_sum + n
            maxlens[b] = max(cur_max, n)

            # flush if reached budget or too many samples
            flush = False
            if sums[b] >= self.tokens_per_batch:
                flush = True
            if len(cur) >= self.max_samples_per_batch:
                flush = True
            if self.padded_tokens_per_batch > 0:
                padded_est = len(cur) * maxlens[b]
                if padded_est >= self.padded_tokens_per_batch:
                    flush = True

            if flush:
                batch = buckets[b]
                buckets[b] = []
                sums[b] = 0
                maxlens[b] = 0
                yield collate_pad(batch, patch_samples=self.patch_samples, hop_samples=self.hop_samples)


def build_webdataset(
    shards: List[str],
    cache_dir: Optional[str],
    shard_shuffle: int,
    sample_shuffle: int,
    base_seconds: int,
    split_long_prob: float,
    max_tokens: int,
    patch_samples: int,
    hop_samples: int,
    enable_channel_grouping: bool,
    limit_num_samples: int = 0,
) -> Iterable[Dict[str, Any]]:
    if wds is None:
        raise RuntimeError("webdataset is not installed. pip install webdataset")

    # (간단) cache: 샤드 리스트를 cache로 복사해 경로를 바꿔치기
    # NOTE: 실제 운영에서는 on-demand/LRU로 바꾸는 걸 추천.
    if cache_dir is not None and len(shards) > 0 and os.path.isfile(shards[0]):
        cached_shards = [cache_shard_to_ssd(s, cache_dir) for s in shards]
    else:
        cached_shards = shards

    ds = wds.WebDataset(cached_shards, resampled=True, handler=wds.ignore_and_continue)
    ds = ds.shuffle(shard_shuffle)

    # decode explicitly (robust)
    ds = ds.decode(wds.numpy_loads, wds.json_loads)

    ds = ds.map(decode_sample)

    # 60/30 -> 10 split
    ds = ds.flatmap(lambda ex: maybe_split_long_to_base(ex, base_seconds=base_seconds, split_prob=split_long_prob))

    # split 이후 shuffle(연속 chunk 방지)
    ds = ds.shuffle(sample_shuffle)

    def _prep(ex):
        if enable_channel_grouping:
            ex = maybe_group_channels(ex, max_tokens=max_tokens, patch_samples=patch_samples, hop_samples=hop_samples)
        else:
            eeg = ex["eeg"]
            C, T = eeg.shape
            P_t = compute_num_patches(T, patch_samples=patch_samples, hop_samples=hop_samples)
            ex["n_tokens"] = int(C * P_t)
            ex["n_patches"] = int(P_t)
            ex["n_channels"] = int(C)
        return ex

    ds = ds.map(_prep)

    if limit_num_samples and limit_num_samples > 0:
        ds = ds.take(limit_num_samples)

    return ds
