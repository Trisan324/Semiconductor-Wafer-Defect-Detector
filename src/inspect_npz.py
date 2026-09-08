import numpy as np

NPZ_PATH = "data/processed/processed_wafer_dataset.npz"  # adjust if needed

data = np.load(NPZ_PATH, allow_pickle=True)

print(f"File: {NPZ_PATH}")
print(f"Arrays found: {list(data.keys())}")
print()

for key in data.keys():
    arr = data[key]
    print(f"- {key}: shape={arr.shape}, dtype={arr.dtype}")

    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] <= 10):
        # small enough to peek at, e.g. labels
        print(f"    sample values: {arr[:3]}")