import wandb
import pandas as pd

# 1. WandB API 초기화 및 Run 객체 가져오기
api = wandb.Api()
# 경로 형식: "엔티티(유저명)/프로젝트명/런_ID"
run = api.run("minsukim207/EEG_FM/ehq45lyz")

# 2. 히스토리 데이터 가져오기 (pandas dataframe)
# 'train_loss' 부분은 WandB에 로깅한 실제 loss 변수명으로 변경하세요.
history = run.history(keys=["_step", "train/l1_mean"], pandas=True)

# 데이터 확인
print(history.head())

def smooth_curve(scalars, weight=0.85):
    """
    WandB와 동일한 방식의 EMA 스무딩 함수
    weight 값이 1에 가까울수록 더 부드러워집니다. (추천: 0.8~0.9)
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        if pd.isna(point): # 결측치 처리
            smoothed.append(last)
            continue
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# 스무딩 적용 (history 데이터프레임에 새로운 열 추가)
history['smoothed_loss'] = smooth_curve(history['train/l1_mean'].values, weight=0.85)

import matplotlib.pyplot as plt
import seaborn as sns

# 논문용 깔끔한 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'serif',            # 폰트 스타일을 명조(Serif) 계열로 설정
    'font.serif': ['Times New Roman'], # 1순위 폰트를 Times New Roman으로 지정
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'pdf.fonttype': 42, # 폰트가 PDF에 내장되도록 설정 (논문 제출 필수 요건)
    'ps.fonttype': 42
})
# plt.rcParams.update({
#     'font.family': 'serif',            # 폰트 스타일을 명조(Serif) 계열로 설정
#     'font.serif': ['Times New Roman'], # 1순위 폰트를 Times New Roman으로 지정
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 14,
#     'legend.fontsize': 12,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'pdf.fonttype': 42, # 폰트가 PDF에 내장되도록 설정 (논문 제출 필수 요건)
#     'ps.fonttype': 42
# })

fig, ax = plt.subplots(figsize=(8, 5))

# 1. Raw Data Plot (연하고 투명하게 배경으로 깔기)
ax.plot(history['_step'], history['train/l1_mean'], 
        color='#4C72B0', alpha=0.4, label='Raw Loss')

# 2. Smoothed Data Plot (진하게 강조)
ax.plot(history['_step'], history['smoothed_loss'], 
        color='#4C72B0', linewidth=2, label='Smoothed Loss (EMA)')

# 축 및 레이블 설정
ax.set_xlabel('Training Steps')
ax.set_ylabel('Pretraining Loss')
# ax.set_title('JEPA Pretraining Loss Convergence')

# X축 포맷 설정 (예: 47000 -> 47k 로 표시되게끔 변경하려면 아래 코드 사용)
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.EngFormatter())

# 레전드 위치 설정
ax.legend(loc='upper right')

# 여백 조정 및 고해상도 PDF로 저장
plt.tight_layout()
plt.savefig('pretraining_loss_curve.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()