import os
import glob
import re
import warnings
import numpy as np
import scipy.signal as signal
import scipy.io as sio  # [NEW] mat 파일용
import pandas as pd     # [NEW] tsv/csv 파일용
import math
import mne
import webdataset as wds
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================================
# [설정 영역] 사용자의 환경에 맞게 수정하세요.
# ==============================================================================
CONFIG = {
    # --------------------------------------------------------------------------
    # [NEW] 파일 형식 및 데이터 소스 설정
    # --------------------------------------------------------------------------
    "ROOT_DIR": "D:\\start\\eeg_foundation_model\\datasets\\Inria Large\\Signals",           # 데이터가 있는 최상위 폴더
    "OUTPUT_PATTERN": "D:\\start\\eeg_foundation_model\\zenodo_InriaLarge/eeg-%06d.tar", # 결과 파일 패턴

    "montage": "standard_1005", # 채널 좌표 매핑을 위한 몽타주 이름 (MNE에서 지원하는 몽타주 사용 권장)
    # "montage": "biosemi64", # 채널 좌표 매핑을 위한 몽타주 이름 (MNE에서 지원하는 몽타주 사용 권장)
    
    # 처리할 파일 확장자 (".edf", ".set", ".mat", ".tsv", ".csv", ".txt", ".bdf", ".vhdr" 등)
    "FILE_EXT": "*.gdf", 

    # [중요] EDF가 아닌 파일(MAT, TSV)을 위한 강제 설정
    # EDF 파일은 아래 두 설정이 무시됩니다 (파일 헤더 정보 사용).
    "FORCE_SFREQ": 512,   # 원본 데이터의 샘플링 레이트 (Hz)
    
    # MAT/TSV 데이터의 채널 순서대로 이름 지정 (Standard-1005 몽타주 이름 권장)
    # 데이터의 열(Column) 또는 행(Row) 순서와 일치해야 합니다.
    "FIXED_CHANNELS": [
        'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4',
        'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5',
        'CP1', 'CP3', 'CP5', 'P3'
    ],

    # [MAT 파일 전용] 데이터가 들어있는 변수 키 (예: 'data', 'X', 'eeg_data')
    "MAT_KEY": "data",

    # [TSV/CSV 파일 전용] 구분자 (탭: '\t', 쉼표: ',')
    "SEPARATOR": "\t", 
    "HAS_HEADER": False, # 파일 첫 줄에 헤더(이름)가 있으면 True, 없으면 False

    # --------------------------------------------------------------------------
    # 전처리 파라미터 (공통)
    # --------------------------------------------------------------------------
    "TARGET_SR": 200,        # 목표 샘플링 레이트
    "BANDPASS": (0.5, 75.0), # (Low cut, High cut)
    "NOTCH_Q": 30.0,         # Notch Filter Q factor
    "NOTCH_FREQ": 50.0,      # None이면 자동 감지
    "CLIP_LIMIT": 15.0,      # Z-score 후 클리핑

    # 세그멘테이션 설정
    "WINDOW_SECONDS": 10,    # 10초 단위로 자르기
    "DROP_LAST": True,       # 자투리 버림

    # 저장 설정
    "SHARD_MAX_SIZE": 1024 ** 3 * 1, # 1GB
    "SHARD_MAX_COUNT": 100000,
    
    # 병렬 처리 설정
    # "NUM_WORKERS": max(1, cpu_count() - 2)
    "NUM_WORKERS": 2,

    # 논문 에서 권고한 제외 피험자 (기술적 결함)
    "EXCLUDE_SUBS": ["A04", "A09", "A17", "A29", "A41", "B78", "B79"]
}


# ==============================================================================
# 1. GDF 전용 데이터 로더 및 채널 정규화
# ==============================================================================
def load_mi_gdf(file_path):
    try:
        # 피험자 ID 추출 및 제외 확인
        sub_id = os.path.basename(os.path.dirname(file_path))
        if any(ex in sub_id for ex in CONFIG["EXCLUDE_SUBS"]):
            return None

        # GDF 로드 [cite: 427]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        
        # 채널 이름 정규화 (접두사 'EEG ' 제거)
        raw.rename_channels(lambda x: x.replace('EEG ', '').strip())
        
        # 27개 표준 채널만 선택 및 순서 강제 고정 (일관성 핵심)
        if not all(ch in raw.ch_names for ch in CONFIG["FIXED_CHANNELS"]):
            return None
        raw.pick_channels(CONFIG["FIXED_CHANNELS"])
        raw.reorder_channels(CONFIG["FIXED_CHANNELS"])

        # 몽타주 적용 (좌표 추출용)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False)
        
        return raw
    except Exception as e:
        print(f"[Load Error] {file_path}: {e}")
        return None

# ==============================================================================
# 2. 신호 처리 클래스 (50Hz 노치 적용)
# ==============================================================================
class MIPreprocessor:
    def __init__(self, target_sr, bandpass, notch_f, clip):
        self.target_sr = target_sr
        self.bandpass = bandpass
        self.notch_f = notch_f
        self.clip = clip

    def apply(self, data, fs):
        nyq = 0.5 * fs
        # 1. 50Hz 노치 필터 적용 (프랑스 데이터 특화)
        if self.notch_f < nyq:
            b, a = signal.iirnotch(self.notch_f, Q=30.0, fs=fs)
            data = signal.filtfilt(b, a, data, axis=-1)

        # 2. 밴드패스 필터 (0.5 - 75 Hz)
        low, high = self.bandpass
        sos = signal.butter(3, [low / nyq, min(high / nyq, 0.99)], btype='band', output='sos')
        data = signal.sosfiltfilt(sos, data, axis=-1)

        # 3. 다운샘플링 (200Hz)
        if fs != self.target_sr:
            gcd = math.gcd(int(fs), int(self.target_sr))
            data = signal.resample_poly(data, self.target_sr // gcd, fs // gcd, axis=-1)

        # 4. 통계적 정규화 (Z-score)
        data = (data - np.mean(data, axis=-1, keepdims=True)) / (np.std(data, axis=-1, keepdims=True) + 1e-8)
        return np.clip(data.astype(np.float16), -self.clip, self.clip)

# ==============================================================================
# 3. 통합 처리 프로세스
# ==============================================================================
def process_single_file(file_path):
    raw = load_mi_gdf(file_path)
    if raw is None: return None

    # 전처리 적용
    preprocessor = MIPreprocessor(CONFIG["TARGET_SR"], CONFIG["BANDPASS"], CONFIG["NOTCH_FREQ"], CONFIG["CLIP_LIMIT"])
    processed_data = preprocessor.apply(raw.get_data(), raw.info['sfreq'])
    
    # 좌표 정보 추출 (float16)
    coords = np.array([raw.info['chs'][i]['loc'][:3] for i in range(len(CONFIG["FIXED_CHANNELS"]))], dtype=np.float16)

    # 세그멘테이션
    samples_list = []
    win_size = int(CONFIG["WINDOW_SECONDS"] * CONFIG["TARGET_SR"])
    sub_id = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(processed_data.shape[-1] // win_size):
        segments = processed_data[:, i*win_size : (i+1)*win_size]
        samples_list.append({
            "key": f"{sub_id}_{file_name}_seg{i:04d}",
            "eeg": segments,
            "coords": coords,
            "meta": {"sub": sub_id, "task": file_name, "sr": CONFIG["TARGET_SR"]}
        })
    return samples_list


# ==============================================================================
# 5. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    out_path = CONFIG["OUTPUT_PATTERN"]
    if out_path.startswith("file:"):
        out_path = out_path.replace("file:", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # [NEW] 설정된 확장자에 따라 파일 검색 (재귀)
    search_pattern = os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"])
    target_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Workers: {CONFIG['NUM_WORKERS']}")
    print(f"Target Pattern: {CONFIG['FILE_EXT']}")
    print(f"Found Files: {len(target_files)}")
    
    writer = wds.ShardWriter(
        "file:" + out_path, 
        maxsize=CONFIG["SHARD_MAX_SIZE"], 
        maxcount=CONFIG["SHARD_MAX_COUNT"]
    )

    print(f"Starting processing... (Window: {CONFIG['WINDOW_SECONDS']}s)")
    
    total_segments = 0
    
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        for results in tqdm(pool.imap_unordered(process_single_file, target_files), total=len(target_files)):
            if results is None:
                continue
            
            for sample in results:
                writer.write({
                    "__key__": sample["key"],
                    "eeg.npy": sample["eeg"],
                    "coords.npy": sample["coords"],
                    "info.json": sample["meta"]
                })
                total_segments += 1

    writer.close()
    print(f"Done. Total segments saved: {total_segments}")
    total_hours = (total_segments * CONFIG["WINDOW_SECONDS"]) / 3600
    print(f"Total EEG hours in dataset: {total_hours:.2f} hours")