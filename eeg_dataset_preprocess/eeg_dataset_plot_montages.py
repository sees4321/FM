import mne
import matplotlib.pyplot as plt
import numpy as np

# 1. 비교할 몽타주 세팅
montage_configs = [
    {"name": "biosemi256", "color": "red", "marker": "o", "alpha": 0.8, "offset": (0.0, -0.007, 0.033)}, 
    {"name": "standard_1005", "color": "blue", "marker": "^", "alpha": 0.3, "offset": (0.001, -0.017, 0.021)}, 
    {"name": "GSN-HydroCel-129", "color": "green", "marker": "*", "alpha": 0.3, "offset": (0.000, -0.002, 0.018)}, 
]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cfg in montage_configs:
    # 몽타주 객체 생성
    montage = mne.channels.make_standard_montage(cfg["name"])
    
    # 채널 좌표 추출 (딕셔너리 형태: {'ch_name': array([x, y, z])})
    pos_dict = montage.get_positions()['ch_pos']
    coords = np.array([pos for pos in pos_dict.values() if pos is not None])
    coords *= 10
    # center = coords.mean(axis=0)
    # diff = coords - center
    # norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
    # coords_norm = diff / (norm_diff + 1e-12)  # 정규화된 좌표 (단위 벡터)

    xs = coords[:, 0]
    ys = coords[:, 1] 
    zs = coords[:, 2]

    # 3D 산점도 그리기
    ax.scatter(xs, ys, zs, 
               c=cfg["color"], 
               marker=cfg["marker"], 
               label=cfg["name"], 
               s=50, 
               alpha=cfg["alpha"])

# 축 라벨 및 범례 설정
ax.set_title("EEG Montage Coordinate Comparison", fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend(loc='upper right')

# (선택) 시야각 조정
ax.view_init(elev=20, azim=45) 

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cfg in montage_configs:
    # 몽타주 객체 생성
    montage = mne.channels.make_standard_montage(cfg["name"])
    
    # 채널 좌표 추출 (딕셔너리 형태: {'ch_name': array([x, y, z])})
    pos_dict = montage.get_positions()['ch_pos']
    coords = np.array([pos for pos in pos_dict.values() if pos is not None])
    center = coords.mean(axis=0)
    diff = coords - center
    norm_diff = np.linalg.norm(diff, axis=1, keepdims=True)
    coords_norm = diff / (norm_diff + 1e-12)  # 정규화된 좌표 (단위 벡터)

    xs = coords_norm[:, 0]
    ys = coords_norm[:, 1] 
    zs = coords_norm[:, 2]

    # 3D 산점도 그리기
    ax.scatter(xs, ys, zs, 
               c=cfg["color"], 
               marker=cfg["marker"], 
               label=cfg["name"], 
               s=50, 
               alpha=cfg["alpha"])

# 축 라벨 및 범례 설정
ax.set_title("EEG Montage Coordinate Comparison", fontsize=14)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend(loc='upper right')

# (선택) 시야각 조정
ax.view_init(elev=20, azim=45) 

plt.tight_layout()
plt.show()