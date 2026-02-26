import time 
from moabb.datasets import Stieger2021

# 데이터 로드
dataset = Stieger2021()
start_time = time.time()
dataset.download()
elapsed = time.time() - start_time
hour, min = divmod(elapsed, 3600)
print(f"dataset download time: {hour:.0f}h {min:.2f} min")