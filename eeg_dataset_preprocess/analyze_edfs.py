import os
from pathlib import Path
from collections import defaultdict
import mne
from tqdm import tqdm

def analyze_edf_folder(folder_path):
    # 폴더 내의 모든 edf 파일 찾기 (하위 폴더 포함)
    edf_files = list(Path(folder_path).rglob('*.edf'))
    total_files = len(edf_files)
    
    if total_files == 0:
        print("지정된 폴더에 EDF 파일이 없습니다.")
        return

    total_duration_sec = 0.0
    
    # 설정이 다른 파일들을 추적하기 위한 딕셔너리
    sfreq_configs = defaultdict(list)
    channel_configs = defaultdict(list)
    
    # MNE의 불필요한 로그 출력 방지
    mne.set_log_level('ERROR')
    
    # tqdm으로 진행상황 표시하며 파일 순회
    for file in tqdm(edf_files, desc="EDF 파일 분석 중", unit="file"):
        try:
            # preload=False를 통해 메모리에 데이터를 올리지 않고 헤더만 빠르게 읽음
            raw = mne.io.read_raw_edf(file, preload=False)
            
            # 1. 시간 길이 계산 (초 단위)
            duration = raw.n_times / raw.info['sfreq']
            total_duration_sec += duration
            
            # 2. Sampling Frequency (s_freq)
            sfreq = raw.info['sfreq']
            sfreq_configs[sfreq].append(file.name)
            
            # 3. 채널 구성 (튜플로 변환하여 딕셔너리 키로 사용)
            ch_names = tuple(raw.ch_names)
            channel_configs[ch_names].append(file.name)
            
        except Exception as e:
            print(f"\n[오류] {file.name} 파일을 읽는 중 문제가 발생했습니다: {e}")

    # ================= 결과 출력 =================
    print("\n\n" + "="*50)
    print("📊 EDF 데이터 통계 요약")
    print("="*50)
    
    # 전체 시간 출력
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = total_duration_sec % 60
    print(f"총 분석 파일 수: {total_files}개")
    print(f"전체 시간 길이: {hours}시간 {minutes}분 {seconds:.2f}초 (총 {total_duration_sec:.2f}초)")
    
    print("-" * 50)
    
    # s_freq 통계 출력
    print("[ Sampling Frequency (s_freq) ]")
    if len(sfreq_configs) == 1:
        sfreq = list(sfreq_configs.keys())[0]
        print(f"✅ 모든 파일이 동일합니다: {sfreq} Hz (100.0%)")
    else:
        print("⚠️ s_freq가 다른 파일들이 존재합니다!")
        # 파일 개수가 많은 순으로 정렬
        sorted_sfreq = sorted(sfreq_configs.items(), key=lambda x: len(x[1]), reverse=True)
        for freq, files in sorted_sfreq:
            count = len(files)
            percentage = (count / total_files) * 100
            print(f"  - {freq} Hz : {count}개 파일 ({percentage:.1f}%)")
            
    print("-" * 50)
    
    # 채널 구성 통계 출력
    print("[ Channel Configuration ]")
    if len(channel_configs) == 1:
        chs = list(channel_configs.keys())[0]
        print(f"✅ 모든 파일이 동일한 채널 구성을 가집니다 (총 {len(chs)}개 채널, 100.0%).")
    else:
        print("⚠️ 채널 구성이 다른 파일들이 존재합니다!")
        # 파일 개수가 많은 순으로 정렬
        sorted_channels = sorted(channel_configs.items(), key=lambda x: len(x[1]), reverse=True)
        for chs, files in sorted_channels:
            count = len(files)
            percentage = (count / total_files) * 100
            print(f"  - {len(chs)}개 채널 구성 (예: {files[0]}) : {count}개 파일 ({percentage:.1f}%)")
            
    print("="*50)

# 실행 예시 (여기에 실제 폴더 경로를 입력하세요)
if __name__ == "__main__":
    target_folder = "D:/open_eeg/tuev"  # 분석할 폴더 경로로 변경하세요.
    analyze_edf_folder(target_folder)
    target_folder = "D:\\main\\coding\\data\\TUAB"
    analyze_edf_folder(target_folder)