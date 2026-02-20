import os
import glob
import re
import warnings
import numpy as np
import scipy.signal as signal
import scipy.io as sio
import pandas as pd
import math
import mne
import webdataset as wds
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================================
# [설정 영역] ICARE 데이터셋 맞춤 설정
# ==============================================================================
CONFIG = {
    # --------------------------------------------------------------------------
    # 경로 및 파일 패턴 설정
    # --------------------------------------------------------------------------
    # 데이터가 위치한 최상위 폴더 경로를 지정하세요.
    "ROOT_DIR": "D:\\SynologyDrive\\open_eeg\\physionet.org\\files\\i-care\\2.1\\training",           
    
    # 결과가 저장될 경로 및 패턴
    "OUTPUT_PATTERN": "D:\\open_eeg_pp\\icare\\eeg-%06d.tar", 
    
    # [중요] ICARE 파일명 규칙: [subject]_[session]_[run]_[modality].mat
    # Modality가 EEG인 파일만 찾기 위해 패턴을 구체화합니다.
    "FILE_EXT": "*EEG.mat", 

    # --------------------------------------------------------------------------
    # 데이터 포맷 강제 설정 (MAT 파일용)
    # --------------------------------------------------------------------------
    "FORCE_SFREQ": 500,   # ICARE Sampling Rate
    
    # [중요] ICARE 데이터의 채널 순서 (19 Channels)
    # T3->T7, T5->P7 등은 아래 로직에서 좌표 매핑을 위해 자동 변환됩니다.
    "FORCE_CH_NAMES": [
        "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", 
        "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz"
    ],
    # 참고: 사용자 입력 중 Fpz, Oz, F9는 19개 카운트에 포함되지 않아 제외했습니다.
    # 만약 데이터 행이 22개라면 이 리스트 뒤에 ["Fpz", "Oz", "F9"]를 추가해야 합니다.

    # [MAT 파일 전용] 데이터가 들어있는 변수 키
    "MAT_KEY": "val",

    # --------------------------------------------------------------------------
    # 전처리 파라미터
    # --------------------------------------------------------------------------
    "TARGET_SR": 200,        # 학습에 사용할 목표 샘플링 레이트
    "BANDPASS": (0.5, 75.0), # Bandpass Filter
    "NOTCH_Q": 30.0,
    "CLIP_LIMIT": 15.0,

    # 세그멘테이션 설정 (Self-supervised Learning용)
    "WINDOW_SECONDS": 60,    # 10초 단위 자르기
    "DROP_LAST": True,

    # 저장 설정
    "SHARD_MAX_SIZE": 1024 ** 3 * 1, # 1GB 단위로 Shard 분할
    "SHARD_MAX_COUNT": 100000,
    
    # 병렬 처리 설정
    "NUM_WORKERS": 4 # CPU 코어 수에 맞춰 조절
}

# ==============================================================================
# 1. 데이터 로더
# ==============================================================================
def load_data_to_mne(file_path):
    """
    ICARE (.mat) 파일을 읽어 MNE Raw 객체로 변환합니다.
    """
    try:
        # MAT 파일 로드
        try:
            mat = sio.loadmat(file_path)
        except NotImplementedError:
            # v7.3 이상의 mat 파일일 경우 h5py나 mat73 등이 필요할 수 있음
            # ICARE는 보통 v7 이하이므로 sio로 가능
            print(f"[Error] Failed to load .mat (check version): {file_path}")
            return None

        key = CONFIG["MAT_KEY"]
        if key not in mat:
            print(f"[Skip] Key '{key}' not found in {os.path.basename(file_path)}")
            return None
            
        data = mat[key] # (Channels, Time) expected

        # 차원 확인 및 전치 (Transpose)
        # ICARE는 보통 (19, Time) 형태로 제공됨
        target_n_ch = len(CONFIG["FORCE_CH_NAMES"])
        shape = data.shape
        
        # 데이터가 (Time, Channels) 형태라면 전치
        if shape[0] != target_n_ch and shape[1] == target_n_ch:
             data = data.T
        
        # 채널 개수 검증
        if data.shape[0] != target_n_ch:
            # 데이터 채널 수와 설정된 채널 수가 다를 경우
            # 데이터가 더 많으면 자르고, 적으면 에러 처리
            if data.shape[0] > target_n_ch:
                data = data[:target_n_ch, :] # 앞에서부터 19개만 사용
            else:
                print(f"[Skip] Channel mismatch: Expected {target_n_ch}, got {data.shape[0]}")
                return None

        # Info 객체 생성
        info = mne.create_info(
            ch_names=CONFIG["FORCE_CH_NAMES"], 
            sfreq=CONFIG["FORCE_SFREQ"], 
            ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        return raw

    except Exception as e:
        print(f"[Load Error] {file_path}: {e}")
        return None

# ==============================================================================
# 2. EEG 전처리 로직 (SmartEEGPreprocessor)
# ==============================================================================
class SmartEEGPreprocessor:
    def __init__(self, target_sr, bandpass_freq, clip_limit):
        self.target_sr = target_sr
        self.bandpass_freq = bandpass_freq
        self.clip_limit = clip_limit

    def detect_line_noise(self, eeg_data, fs):
        try:
            freqs, psd = signal.welch(eeg_data, fs, nperseg=int(fs), axis=-1)
            mean_psd = np.mean(psd, axis=0)
            
            idx_50 = np.argmin(np.abs(freqs - 50))
            idx_60 = np.argmin(np.abs(freqs - 60))
            
            if idx_50 >= len(mean_psd) or idx_60 >= len(mean_psd): return None

            power_50 = mean_psd[idx_50]
            power_60 = mean_psd[idx_60]

            if power_60 > power_50 * 1.5: return 60.0
            elif power_50 > power_60 * 1.5: return 50.0
            return None
        except:
            return None

    def apply(self, eeg_data, original_sr):
        nyq = 0.5 * original_sr
        low_cut = self.bandpass_freq[0]
        high_cut = self.bandpass_freq[1]

        if high_cut >= nyq:
            adjusted_high = nyq - 1.0 
            if adjusted_high <= low_cut: adjusted_high = nyq - 0.1
        else:
            adjusted_high = high_cut

        # 1. Notch Filter (Line noise removal)
        line_freq = self.detect_line_noise(eeg_data, original_sr)
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=CONFIG["NOTCH_Q"], fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        # 2. Bandpass Filter
        Wn_low = low_cut / nyq
        Wn_high = adjusted_high / nyq
        if Wn_high >= 1.0: Wn_high = 0.99 
        
        sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        # 3. Resample
        if original_sr != self.target_sr:
            # poly resampling이 더 정확하고 빠름
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            up = int(self.target_sr // gcd)
            down = int(original_sr // gcd)
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=-1)

        # 4. Z-score Normalization
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)

        # 5. Clipping & Type Casting
        eeg_data = np.clip(eeg_data, -self.clip_limit, self.clip_limit)
        eeg_data = eeg_data.astype(np.float16) # 용량 절약

        return eeg_data

# ==============================================================================
# 3. 채널 이름 정규화 및 좌표 추출
# ==============================================================================
def clean_channel_names(raw):
    mapping = {}
    for ch_name in raw.ch_names:
        # 공백 제거 및 대문자 변환
        clean_name = ch_name.strip().upper()
        
        # ICARE/Standard 10-20 구형 명칭 -> MNE Standard 1005 명칭 매핑
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8',  # T3, T4는 보통 T7, T8로 매핑됨
            'T5': 'P7', 'T6': 'P8',  # T5, T6는 보통 P7, P8로 매핑됨
        }
        final_name = name_map.get(clean_name, clean_name.capitalize()) 
        
        if ch_name != final_name:
            mapping[ch_name] = final_name

    # 이름 변경 적용 (중복 방지 로직 포함)
    final_mapping = {k: v for k, v in mapping.items() if v not in raw.ch_names or k == v}
    try:
        raw.rename_channels(final_mapping)
    except: pass
    
    return raw

def get_valid_channel_indices(raw):
    valid_names = []
    valid_coords = []
    
    # 몽타주 정보에서 좌표 추출
    for ch_name in raw.ch_names:
        if ch_name not in raw.info['chs'][raw.ch_names.index(ch_name)]['ch_name']:
            continue
            
        ch_idx = raw.ch_names.index(ch_name)
        loc = raw.info['chs'][ch_idx]['loc'][:3] # x, y, z 좌표
        
        # 좌표가 유효한지 확인 (0,0,0이 아니고 NaN이 아님)
        if not np.all(np.isnan(loc)) and not np.all(loc == 0):
            valid_names.append(ch_name)
            valid_coords.append(loc)
            
    return valid_names, valid_coords

# ==============================================================================
# 4. 파일 처리 Worker
# ==============================================================================
def process_single_file(file_path):
    try:
        # 1. 데이터 로드
        raw = load_data_to_mne(file_path)
        if raw is None: return None

        if raw._data.dtype == np.float64:
             raw._data = raw._data.astype(np.float32)

        # 2. 채널 이름 표준화 (T3 -> T7 등 변환)
        raw = clean_channel_names(raw)

        # 3. 몽타주(좌표) 적용
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
        except: 
            print(f"[Warning] Montage failed for {file_path}")
            pass
        
        # 4. 좌표가 유효한 채널만 필터링
        valid_names, valid_coords = get_valid_channel_indices(raw)
        
        # 최소 채널 수 확인 (예: 15개 미만이면 드랍)
        if len(valid_names) < 15:
            # print(f"[Skip] Not enough valid channels: {len(valid_names)}")
            return None

        raw.pick(valid_names)

        # 5. 전처리 및 세그멘테이션
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        preprocessor = SmartEEGPreprocessor(
            target_sr=CONFIG["TARGET_SR"],
            bandpass_freq=CONFIG["BANDPASS"],
            clip_limit=CONFIG["CLIP_LIMIT"]
        )
        
        processed_full = preprocessor.apply(data, sfreq)

        # 윈도우 커팅 (Segmentation)
        samples_list = []
        window_samples = int(CONFIG["WINDOW_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]
        
        if total_length < window_samples: 
            return None
        
        # 파일명에서 메타 정보 추출
        # 예: Sub01_Sess01_Run01_EEG.mat -> Sub01_Sess01_Run01
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        num_segments = total_length // window_samples
        coords_array = np.array(valid_coords, dtype=np.float16)
        
        for i in range(num_segments):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            
            segment_data = processed_full[:, start_idx:end_idx]
            
            # 고유 키 생성 (파일명_세그먼트번호)
            key = f"{file_name}_seg{i:04d}"
            
            sample_dict = {
                "key": key,
                "eeg": segment_data,
                "coords": coords_array,
                "meta": {
                    "source": file_name,
                    "segment_idx": i
                    # SSL용이라 최소한의 메타만 저장
                }
            }
            samples_list.append(sample_dict)
            
        return samples_list

    except Exception as e:
        print(f"[Error Processing] {file_path}: {e}")
        return None

# ==============================================================================
# 5. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    out_path = CONFIG["OUTPUT_PATTERN"]
    
    # 윈도우 경로 호환성을 위해 체크
    if out_path.startswith("file:"):
        out_path = out_path.replace("file:", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ICARE 데이터셋 검색 (재귀적)
    search_pattern = os.path.join(CONFIG["ROOT_DIR"], "**", CONFIG["FILE_EXT"])
    target_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Workers: {CONFIG['NUM_WORKERS']}")
    print(f"Looking for: {CONFIG['FILE_EXT']}")
    print(f"Found Files: {len(target_files)}")
    
    if len(target_files) == 0:
        print("No files found. Please check ROOT_DIR.")
        exit()

    # WebDataset Writer 생성
    writer = wds.ShardWriter(
        'file:' + out_path, 
        maxsize=CONFIG["SHARD_MAX_SIZE"], 
        maxcount=CONFIG["SHARD_MAX_COUNT"]
    )

    print(f"Starting processing... (Target SR: {CONFIG['TARGET_SR']}Hz)")
    
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