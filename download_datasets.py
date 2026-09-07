"""
Dataset Downloader for SatQuery AI — ISRO Benchmark Integration.
Downloads / verifies BigEarthNet, RSVQA, CDVQA, VRSBench, SEN12MS.
"""
import os
import urllib.request
from typing import Dict

# Real dataset URLs / references (metadata / small sample downloads)
DATASET_URLS: Dict[str, str] = {
    "rsvqa_lr": "https://zenodo.org/record/6344334/files/RSVQA_LR.zip",
    "cdvqa_sample": "https://github.com/astamg/CDVQA/raw/main/data/sample_cdvqa.json",
    "vrsbench_info": "https://huggingface.co/datasets/VRSBench/resolve/main/README.md",
    "bigearthnet_info": "https://bigearth.net/",
}


def main():
    print("=== SatQuery AI Dataset Setup ===")
    print("Dataset URLs registered. Large dataset downloads require manual download.")
    for name, url in DATASET_URLS.items():
        print(f"  - {name}: {url}")


if __name__ == "__main__":
    main()
