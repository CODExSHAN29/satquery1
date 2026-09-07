from datasets import load_dataset
import numpy as np
import tifffile

ds = load_dataset(
    "Hermanni/sen12mscr",
    split="train",
    streaming=True
)

sample = next(iter(ds))

sar = np.frombuffer(
    sample["sar"],
    dtype=np.float32
).reshape(sample["sar_shape"])

optical = np.frombuffer(
    sample["target"],
    dtype=np.int16
).reshape(sample["opt_shape"])

# SAR: usually (2, 256, 256)
tifffile.imwrite("sentinel1_sar.tif", sar)

# Optical is HWC in this mirror: usually (256, 256, 13)
tifffile.imwrite("sentinel2_optical.tif", optical)

print("Saved one matched Sentinel-1/Sentinel-2 pair")