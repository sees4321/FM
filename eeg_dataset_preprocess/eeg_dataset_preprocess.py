import os
import glob
import re
import warnings
import numpy as np
import scipy.signal as signal
import math
import mne
import webdataset as wds
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================================
# [설정 영역] 사용자의 환경에 맞게 수정하세요.
# ==============================================================================
CONFIG = {
    # 입력/출력 경로
    "ROOT_DIR": "D:/open_eeg_/sleep-edfx/eeg",         # 원본 EDF 파일들이 있는 최상위 폴더
    "OUTPUT_PATTERN": "D:/open_eeg_pp/sleepedf/eeg-%06d.tar", # 저장될 파일 패턴 (폴더 자동 생성됨)
    "FILE_EXT": ".edf",    # 처리할 파일 확장자
    
    # 전처리 파라미터
    "TARGET_SR": 200,        # 목표 샘플링 레이트
    "BANDPASS": (0.5, 75.0),# (Low cut, High cut)
    "NOTCH_Q": 30.0,         # Notch Filter Q factor
    "CLIP_LIMIT": 15.0,      # Z-score 후 클리핑 (표준편차 배수)

    # [NEW] 세그멘테이션 설정
    "WINDOW_SECONDS": 30,    # 10초 단위로 자르기
    "DROP_LAST": True,       # 마지막에 10초 안 되는 자투리는 버림

    # 저장 설정
    "SHARD_MAX_SIZE": 1024 ** 3 * 1, # 1GB 단위로 파일 나눔
    "SHARD_MAX_COUNT": 100000,       # 파일당 최대 샘플 수 (넉넉하게 설정)
    
    # 병렬 처리 설정
    "NUM_WORKERS": max(1, cpu_count() - 2) # CPU 코어 수 - 2 (시스템 멈춤 방지)
}

# ==============================================================================
# 1. EEG 전처리 로직 (Signal Processing)
# ==============================================================================
class SmartEEGPreprocessor:
    def __init__(self, target_sr, bandpass_freq, clip_limit):
        self.target_sr = target_sr
        self.bandpass_freq = bandpass_freq
        self.clip_limit = clip_limit

    def detect_line_noise(self, eeg_data, fs):
        """
        PSD를 분석하여 50Hz 또는 60Hz 중 더 강한 노이즈를 찾습니다.
        """
        # 속도를 위해 nperseg를 fs로 설정 (1Hz 해상도)
        try:
            freqs, psd = signal.welch(eeg_data, fs, nperseg=int(fs), axis=-1)
            mean_psd = np.mean(psd, axis=0)
            
            # 인덱스 보호 처리
            idx_50 = np.argmin(np.abs(freqs - 50))
            idx_60 = np.argmin(np.abs(freqs - 60))
            
            if idx_50 >= len(mean_psd) or idx_60 >= len(mean_psd):
                return None

            power_50 = mean_psd[idx_50]
            power_60 = mean_psd[idx_60]

            # 1.5배 이상 차이나면 해당 주파수를 노이즈로 간주
            if power_60 > power_50 * 1.5: return 60.0
            elif power_50 > power_60 * 1.5: return 50.0
            return None
        except:
            return None

def apply(self, eeg_data, original_sr):
        # [중요] 1. Bandpass Filter 설정 자동 보정
        nyq = 0.5 * original_sr
        low_cut = self.bandpass_freq[0]
        high_cut = self.bandpass_freq[1]

        # 만약 원본 데이터가 100Hz라서 Nyq가 50Hz인데, High Cut이 100Hz라면?
        # -> Scipy 에러 발생함. 따라서 High Cut을 Nyquist보다 조금 낮게 강제 조정.
        if high_cut >= nyq:
            # 예: 50Hz Nyq라면 48Hz까지만 필터링 (여유분 둠)
            adjusted_high = nyq - 1.0 
            # 만약 low_cut보다도 작아지면(데이터가 너무 저품질), 필터링 스킵 고려해야 함
            if adjusted_high <= low_cut:
                # 데이터 품질이 너무 낮음 -> 필터링 없이 진행하거나 Low cut만 적용
                # 여기서는 안전하게 High Cut을 Nyq - epsilon으로 설정
                adjusted_high = nyq - 0.1
        else:
            adjusted_high = high_cut

        # 2. Notch Filter (원본 SR 기준)
        # 100Hz 데이터(Nyq 50)라면 60Hz 노이즈는 물리적으로 존재 불가능 -> 자동 스킵됨
        line_freq = self.detect_line_noise(eeg_data, original_sr)
        if line_freq and line_freq < nyq:
            b_notch, a_notch = signal.iirnotch(line_freq, Q=30.0, fs=original_sr)
            eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data, axis=-1)

        # 3. Bandpass Filter 적용
        # 보정된 adjusted_high 사용
        Wn_low = low_cut / nyq
        Wn_high = adjusted_high / nyq
        
        # 안전장치: Wn 범위 체크 (0 < Wn < 1)
        if Wn_high >= 1.0: Wn_high = 0.99 
        
        sos = signal.butter(3, [Wn_low, Wn_high], btype='band', analog=False, output='sos')
        eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=-1)

        # 4. Resample (100Hz -> 200Hz 업샘플링 수행)
        if original_sr != self.target_sr:
            gcd = math.gcd(int(original_sr), int(self.target_sr))
            up = int(self.target_sr // gcd)
            down = int(original_sr // gcd)
            eeg_data = signal.resample_poly(eeg_data, up, down, axis=-1)

            # num_samples = int(eeg_data.shape[-1] * self.target_sr / original_sr)
            # eeg_data = signal.resample(eeg_data, num_samples, axis=-1)

        # 5. Z-score Normalization
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        eeg_data = (eeg_data - mean) / (std + 1e-8)

        # 6. Type Casting & Clipping
        eeg_data = eeg_data.astype(np.float16)
        eeg_data = np.clip(eeg_data, -self.clip_limit, self.clip_limit)

        return eeg_data

# ==============================================================================
# 2. 채널 이름 정규화 및 좌표 추출 헬퍼
# ==============================================================================
def clean_channel_names(raw):
    mapping = {}
    for ch_name in raw.ch_names:
        new_name = re.sub(r'(?i)(EEG|[-_]REF|[-_]LE|[-_]MON|[-_]AVG)', '', ch_name).strip()
        new_name = new_name.upper()
        name_map = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'
        }
        final_name = name_map.get(new_name, new_name.capitalize())
        mapping[ch_name] = final_name
    raw.rename_channels(mapping)
    return raw

def get_valid_channel_indices(raw):
    valid_names = []
    valid_coords = []
    for ch_name in raw.ch_names:
        idx = raw.ch_names.index(ch_name)
        loc = raw.info['chs'][idx]['loc'][:3]
        if not np.all(np.isnan(loc)) and not np.all(loc == 0):
            valid_names.append(ch_name)
            valid_coords.append(loc)
    return valid_names, valid_coords

# ==============================================================================
# 3. Worker 함수 (병렬 처리 단위)
# ==============================================================================
def process_single_edf(file_path):
    """
    EDF 파일 하나를 읽어서 전처리 후 결과 Dict를 반환.
    Writer는 메인 프로세스에 있으므로 여기서는 데이터만 리턴함.
    """
    try:
        # 경고 메시지 무시 (MNE 로딩 시 잡다한 경고 많음)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # preload=True 필수 (필터링 위해)
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # [중요] 메모리 최적화: 로드 직후 float32로 변환 (float64 대비 용량 절반)
        if raw._data.dtype == np.float64:
             raw._data = raw._data.astype(np.float32)

        # 1. 채널 이름 정리
        raw = clean_channel_names(raw)
        
        # 2. EEG 채널만 1차 필터링 (EOG, ECG 등 제거)
        raw = clean_channel_names(raw)
        try:
            raw.pick("eeg", exclude="bads")
        except ValueError: return None # EEG 없음
        if len(raw.ch_names) < 3: # 채널 너무 적으면 스킵
            return None

        # 3. Standard-1005 몽타주 적용 (High-density 커버)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False, on_missing='ignore')

        # 4. 좌표 유효성 검사 (좌표 없는 채널 제거)
        valid_names, valid_coords = get_valid_channel_indices(raw)
        
        if len(valid_names) < 3:
            print(f"\n[Warning] {file_path}: Not enough valid channels after coord check.")
            return None # 유효 채널 부족

        # 5. Raw 객체에서 채널 동기화 (Drop channels without coords)
        raw.pick(valid_names)

        # [CRITICAL CHECK] 신호 채널 수와 좌표 수 일치 확인
        if len(raw.ch_names) != len(valid_coords):
            raise RuntimeError(f"Channel Mismatch: Signal={len(raw.ch_names)}, Coords={len(valid_coords)}")

        # 6. 신호 전처리
        data = raw.get_data() # (Channels, Time)
        sfreq = raw.info['sfreq']
        
        preprocessor = SmartEEGPreprocessor(
            target_sr=CONFIG["TARGET_SR"],
            bandpass_freq=CONFIG["BANDPASS"],
            clip_limit=CONFIG["CLIP_LIMIT"]
        )
        
        processed_full = preprocessor.apply(data, sfreq)
        
        # 6. Segmentation (자르기)
        samples_list = []
        
        # 목표 샘플 수 (예: 10초 * 250Hz = 2500 samples)
        window_samples = int(CONFIG["WINDOW_SECONDS"] * CONFIG["TARGET_SR"])
        total_length = processed_full.shape[-1]
        
        if total_length < window_samples:
            return None
        
        file_name = os.path.basename(file_path).replace(".edf", "")
        parent_folder = os.path.basename(os.path.dirname(file_path))
        
        # Sliding Window Loop
        num_segments = total_length // window_samples
        
        # 좌표는 모든 세그먼트가 공유함
        coords_array = np.array(valid_coords, dtype=np.float16)
        if processed_full.shape[-2] != len(coords_array):
            print("Shape mismatch after preprocessing, maybe channel error.")
            return None
        
        for i in range(num_segments):
            start_idx = i * window_samples
            end_idx = start_idx + window_samples
            
            # 잘라내기
            segment_data = processed_full[:, start_idx:end_idx]
            
            # 키 생성 (파일명_seg001)
            key = f"{parent_folder}_{file_name}_seg{i:04d}"
            
            sample_dict = {
                "key": key,
                "eeg": segment_data, # (Ch, Window_Time)
                "coords": coords_array,
                "meta": {
                    "source": file_name,
                    "segment_idx": i,
                    "start_sec": i * CONFIG["WINDOW_SECONDS"],
                    "n_channels": len(valid_names),
                    "ch_names": valid_names
                }
            }
            samples_list.append(sample_dict)
            
        return samples_list # 리스트 반환

    except Exception:
        # traceback.print_exc()
        return None

# ==============================================================================
# 4. 메인 실행부
# ==============================================================================
if __name__ == "__main__":
    out_path = CONFIG["OUTPUT_PATTERN"]
    if out_path.startswith("file:"):
        out_path = out_path.replace("file:", "")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    edf_files = glob.glob(os.path.join(CONFIG["ROOT_DIR"], "**/*.edf"), recursive=True)
    
    # [중요] 메모리 오류가 계속 나면 NUM_WORKERS를 더 줄이세요 (예: 2 또는 4)
    print(f"Workers: {CONFIG['NUM_WORKERS']}") 
    
    writer = wds.ShardWriter(
        "file:" + out_path, 
        maxsize=CONFIG["SHARD_MAX_SIZE"], 
        maxcount=CONFIG["SHARD_MAX_COUNT"]
    )

    print(f"Starting processing... (Window: {CONFIG['WINDOW_SECONDS']}s)")
    
    total_segments = 0
    
    with Pool(CONFIG["NUM_WORKERS"]) as pool:
        # 결과가 이제 List[Dict] 형태로 넘어옴
        for results in tqdm(pool.imap_unordered(process_single_edf, edf_files), total=len(edf_files)):
            if results is None:
                continue
            
            # 리스트 안의 세그먼트들을 하나씩 저장
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