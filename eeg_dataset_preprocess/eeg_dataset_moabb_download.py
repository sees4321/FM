from moabb.datasets import Liu2024, Cho2017, Ofner2017, Schirrmeister2017, Lee2019_SSVEP


dataset = Lee2019_SSVEP()
print(f"Dataset: {dataset.code}")

# 데이터셋 다운로드 (최초 1회 실행 시 시간 소요됨)
print("Checking/Downloading dataset...")
try:
    dataset.download()
except Exception as e:
    print(f"Download Warning (check internet or path): {e}")