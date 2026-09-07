"""Smoke-test mode: --max-samples 100 --epochs 1."""
import argparse, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from satquery_ai.models.sar_optical_fusion import SAROpticalFusion, train_fusion_adapter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(f"[SMOKE TEST] max_samples={args.max_samples}, epochs={args.epochs}")
    model = SAROpticalFusion()
    train_fusion_adapter(model, [], epochs=args.epochs, max_samples=args.max_samples)
    print("[SMOKE TEST] Completed successfully.")

if __name__ == "__main__":
    main()
