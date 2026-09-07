"""Lightweight precomputed embeddings script for SAR+Optical adapter training."""
import argparse
import sys
sys.path.insert(0, "C:/Users/codex/newsat")
from satquery_ai.models.sar_optical_fusion import precompute_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default="datasets/raw/sen12ms")
    parser.add_argument("--output-dir", type=str, default="datasets/embeddings")
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()
    precompute_embeddings(args.dataset_dir, args.output_dir, args.max_samples)

if __name__ == "__main__":
    main()
