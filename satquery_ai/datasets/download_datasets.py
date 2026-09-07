import os
import sys
import urllib.request
import zipfile
import tarfile
from typing import Dict


DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "raw"))

# Official Dataset URLs / Repositories
DATASET_URLS: Dict[str, str] = {
    "rsvqa_lr": "https://zenodo.org/record/6344334/files/RSVQA_LR.zip",
    "cdvqa_sample": "https://github.com/astamg/CDVQA/raw/main/data/sample_cdvqa.json",
    "vrsbench_info": "https://huggingface.co/datasets/VRSBench/resolve/main/README.md",
    "bigearthnet_info": "https://bigearth.net/"
}


def setup_dataset_directories():
    """Creates directory structure for raw and processed datasets."""
    subdirs = ["bigearthnet", "vrsbench", "rsvqa", "cdvqa"]
    for s in subdirs:
        path = os.path.join(DATASET_DIR, s)
        os.makedirs(path, exist_ok=True)
    print(f"[Dataset Setup] Created dataset directories in: {DATASET_DIR}")


def download_file(url: str, output_path: str):
    """Downloads a file from a URL with progress indication."""
    print(f"[Download] Fetching: {url} -> {output_path}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"[Download Successful] Saved to {output_path}")
    except Exception as e:
        print(f"[Download Warning] Could not download from {url}: {e}")


def main():
    print("=== SatQuery AI Dataset Downloader ===")
    setup_dataset_directories()

    # Create dummy sample manifests so training/fine-tuning scripts can run immediately
    create_sample_manifests()


def create_sample_manifests():
    """Generates ready-to-train sample manifests matching official benchmark formats."""
    import json

    # 1. BigEarthNet Sample Manifest
    bigearthnet_samples = [
        {
            "patch_id": "S2A_MSIL2A_20170613T101031_0_45",
            "optical_path": os.path.join(DATASET_DIR, "bigearthnet", "sample_optical.tif"),
            "sar_path": os.path.join(DATASET_DIR, "bigearthnet", "sample_sar.tif"),
            "land_cover": ["Coniferous forest", "Mixed forest", "Water bodies"],
            "caption": "Co-registered Sentinel-1 SAR and Sentinel-2 optical imagery showing forest and water body boundary."
        }
    ]
    with open(os.path.join(DATASET_DIR, "bigearthnet", "manifest.json"), "w") as f:
        json.dump(bigearthnet_samples, f, indent=2)

    # 2. CDVQA (Bi-temporal Change VQA) Sample Manifest
    cdvqa_samples = [
        {
            "pair_id": "CDVQA_001",
            "t1_image": os.path.join(DATASET_DIR, "cdvqa", "t1_sample.tif"),
            "t2_image": os.path.join(DATASET_DIR, "cdvqa", "t2_sample.tif"),
            "question": "What changed between these two dates, and where did the change occur?",
            "answer": "Between T1 and T2, agricultural land was converted into built-up structures in the eastern quadrant.",
            "change_percentage": 14.2
        }
    ]
    with open(os.path.join(DATASET_DIR, "cdvqa", "manifest.json"), "w") as f:
        json.dump(cdvqa_samples, f, indent=2)

    # 3. VRSBench (Grounding & Single-Image VQA) Sample Manifest
    vrsbench_samples = [
        {
            "image_id": "VRS_001",
            "image_path": os.path.join(DATASET_DIR, "vrsbench", "sample_vrs.tif"),
            "question": "Describe the land-cover and major objects visible in this image.",
            "answer": "The patch depicts an urban area intersected by a highway and river.",
            "grounding": {
                "query": "river",
                "box_2d": [100, 150, 400, 480]
            }
        }
    ]
    with open(os.path.join(DATASET_DIR, "vrsbench", "manifest.json"), "w") as f:
        json.dump(vrsbench_samples, f, indent=2)

    print("[Dataset Setup] Created initial benchmark manifests successfully.")


if __name__ == "__main__":
    main()