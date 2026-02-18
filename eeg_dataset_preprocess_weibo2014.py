   
import os
import glob
import numpy as np
import scipy.io as sio  # .mat 파일 로드용
import mne
import webdataset as wds
from tqdm import tqdm

# ==============================================================================
# [설정 영역] 사용자 환경에 맞게 수정하세요
# ==============================================================================
CONFIG = {
    # 1. 입력 .mat 파일들이 있는 폴더 경로 (또는 파일 리스트)
    "INPUT_DIR": "C:\\Users\\user\\mne_data\\MNE-weibo-2014",
    "MAT_FILE_PATTERN": "*.mat",  # 파일 확장자 패턴

    # 2. 출력 경로 설정 (WebDataset 포맷)
    "OUTPUT_PATTERN": "D:/open_eeg_pp/moabb_weibo2014/eeg-%06d.tar",
    
    # 3. 저장 설정
    "SHARD_MAX_SIZE": 1024 ** 3 * 1,  # 1GB 단위로 분할
    "SHARD_MAX_COUNT": 100000,        # 또는 샘플 수 기준으로 분할

    # 4. [중요] 채널 이름 리스트
    # .mat 파일의 데이터(행) 순서와 정확히 일치하는 채널명을 적어야 좌표를 구할 수 있습니다.
    # (예시: 10-20 시스템 표준 명칭 사용)
    "CHANNEL_NAMES": ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
}

# ==============================================================================
# 1. 좌표 추출 헬퍼 함수
# ==============================================================================
def get_electrode_coords(ch_names):
    """
    채널 이름 리스트를 받아 MNE 표준 몽타주(Standard 1005) 기준의 
    3D 좌표(x, y, z) 배열을 반환합니다.
    """
    # 가상의 Info 객체 생성
    info = mne.create_info(ch_names, sfreq=100, ch_types='eeg')
    
    # 표준 몽타주 설정 (없으면 좌표를 못 구함)
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        info.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"Warning: Montage setting failed. Coords will be zeros. {e}")
        return np.zeros((len(ch_names), 3), dtype=np.float16)

    coords_list = []
    valid_chs = 0
    
    for ch in ch_names:
        # 채널 인덱스 찾기
        if ch in info.ch_names:
            ch_idx = info.ch_names.index(ch)
            loc = info['chs'][ch_idx]['loc'][:3] # 앞의 3개가 x,y,z
            
            # 좌표가 없는 경우 (NaN or 0)
            if np.all(np.isnan(loc)) or np.all(loc == 0):
                # print(f"Warning: No coords for {ch}")
                coords_list.append([0.0, 0.0, 0.0])
            else:
                coords_list.append(loc)
                valid_chs += 1
        else:
            coords_list.append([0.0, 0.0, 0.0])

    print(f"Coordinates extraction: {valid_chs}/{len(ch_names)} channels found in standard montage.")
    return np.array(coords_list, dtype=np.float16)

# ==============================================================================
# 2. 메인 변환 로직
# ==============================================================================
def main():
    eeg_chan_mask = [True]*57 + [False] + [True]*3 + [False]*3
    eeg_chan_mask = np.array(eeg_chan_mask)
    # 출력 폴더 생성
    out_path = CONFIG["OUTPUT_PATTERN"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 입력 파일 리스트 확보
    search_path = os.path.join(CONFIG["INPUT_DIR"], CONFIG["MAT_FILE_PATTERN"])
    mat_files = sorted(glob.glob(search_path))
    
    if not mat_files:
        print(f"No .mat files found in {search_path}")
        return

    print(f"Found {len(mat_files)} .mat files.")

    # 1. 채널 좌표 미리 계산 (모든 파일 공통이라고 가정)
    coords_array = get_electrode_coords(CONFIG["CHANNEL_NAMES"])
    
    # 2. WebDataset Writer 초기화
    writer = wds.ShardWriter(
        'file:'+out_path, 
        maxsize=CONFIG["SHARD_MAX_SIZE"], 
        maxcount=CONFIG["SHARD_MAX_COUNT"]
    )

    total_samples = 0

    # 3. 파일 순회
    for mat_path in tqdm(mat_files, desc="Processing files"):
        try:
            # .mat 로드
            mat_data = sio.loadmat(mat_path)
            
            # 'data' 키 확인 (데이터셋마다 키 이름이 다를 수 있음, 필요시 수정)
            if 'data' not in mat_data:
                print(f"Skipping {mat_path}: key 'data' not found.")
                continue
            
            # 데이터 형상: (Channels, Time, Trials) 가정
            eeg_3d = mat_data['data'] 
            
            # 차원 확인 및 보정
            # 만약 (Trials, Channels, Time) 등으로 되어있다면 transpose 필요
            # 여기서는 사용자 요청대로 (Ch, Time, Trials)라고 가정
            n_channels, n_time, n_trials = eeg_3d.shape
            # print(n_channels, len(eeg_chan_mask))
            # 채널 수 검증
            
            filename_base = os.path.splitext(os.path.basename(mat_path))[0]

            # 4. Trial 단위로 저장
            for i in range(n_trials):
                # (Ch, Time) 형태로 추출

                trial_data = eeg_3d[eeg_chan_mask, :, i]
                mean = np.mean(trial_data, axis=-1, keepdims=True)
                std = np.std(trial_data, axis=-1, keepdims=True)
                trial_data = (trial_data - mean) / (std + 1e-8)

                
                # float32 변환 (용량 절약 및 호환성)
                trial_data = trial_data.astype(np.float16)
                trial_data = np.clip(trial_data, -15.0, 15.0)

                # 고유 키 생성
                key = f"{filename_base}_trial{i:04d}"
                if trial_data.shape[0] != len(CONFIG["CHANNEL_NAMES"]):
                    print(f"Warning: {mat_path} has {trial_data.shape[0]} channels, but config has {len(CONFIG['CHANNEL_NAMES'])}. Check consistency.")
                
                # 메타데이터 구성
                meta = {
                    "origin_file": list(os.path.basename(mat_path)), # json serialization을 위해 list 변환 등 주의
                    "trial_index": i,
                    "n_channels": int(n_channels),
                    "n_time": int(n_time)
                }

                # WebDataset 쓰기
                writer.write({
                    "__key__": key,
                    "eeg.npy": trial_data,   # (Ch, Time)
                    "coords.npy": coords_array, # (Ch, 3) - 미리 구해둔 좌표
                    "info.json": meta
                })
                
                total_samples += 1

        except Exception as e:
            print(f"Error processing {mat_path}: {e}")
            continue

    writer.close()
    print("="*50)
    print(f"Conversion Complete.")
    print(f"Total samples saved: {total_samples}")
    print(f"Output location: {os.path.dirname(out_path)}")

if __name__ == "__main__":
    main()