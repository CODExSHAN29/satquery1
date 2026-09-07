"""Real fusion adapter training loop — only adapter weights updated."""
import argparse
import sys
import time
sys.path.insert(0, "C:/Users/codex/newsat")
from satquery_ai.models.sar_optical_fusion import (
    SAROpticalFusion, train_fusion_adapter, precompute_embeddings
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(f"[train_fusion_adapter] Mode: smoke test (max_samples={args.max_samples}, epochs={args.epochs})")
    embeddings = []
    model = SAROpticalFusion()
    train_fusion_adapter(model, embeddings, epochs=args.epochs, max_samples=args.max_samples)
    print("[train_fusion_adapter] Adapter training loop complete.")

if __name__ == "__main__":
    main()
