"""
Evaluation script for Bi-temporal Satellite Image Change Detection.
Evaluates the SatQuery BiTemporalChangePipeline against authentic LEVIR-CD benchmark pairs.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import numpy as np
from PIL import Image
from satquery_ai.models.bitemporal_change_service import BiTemporalChangePipeline, analyze_bitemporal
from satquery_ai.tools.bitemporal_change import BiTemporalChangeTool

def evaluate_pair(pair_dir: str):
    t1_path = os.path.join(pair_dir, 't1_before.png')
    t2_path = os.path.join(pair_dir, 't2_after.png')
    gt_path = os.path.join(pair_dir, 'ground_truth_mask.png')

    # Load Ground Truth
    gt_img = Image.open(gt_path).convert('L')
    gt_arr = (np.array(gt_img) > 128).astype(np.uint8)

    # Run OpenCV pipeline
    pipeline = BiTemporalChangePipeline(threshold=0.15)
    diff, mask, evidence = pipeline.detect_change(t1_path, t2_path)
    pred_arr = (mask > 128).astype(np.uint8)

    # Compute Pixel-Level Evaluation Metrics
    tp = np.sum((pred_arr == 1) & (gt_arr == 1))
    fp = np.sum((pred_arr == 1) & (gt_arr == 0))
    fn = np.sum((pred_arr == 0) & (gt_arr == 1))
    tn = np.sum((pred_arr == 0) & (gt_arr == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    gt_pct = float(np.sum(gt_arr) / gt_arr.size * 100.0)
    pred_pct = float(evidence.get('changed_area_percent', 0.0))

    return {
        'gt_change_percent': round(gt_pct, 2),
        'pred_change_percent': round(pred_pct, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'iou': round(iou, 4),
        'regions_detected': evidence.get('change_regions_count', 0)
    }

def main():
    base_dir = os.path.join('datasets', 'evaluation', 'bitemporal_pairs')
    manifest_file = os.path.join(base_dir, 'manifest.json')

    if not os.path.exists(manifest_file):
        print(f"Manifest not found at {manifest_file}")
        return

    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    print("=" * 70)
    print(" SATQUERY AI — BITEMPORAL CHANGE DETECTION BENCHMARK EVALUATION")
    print(" Dataset: LEVIR-CD Benchmark (0.5m/px VHR Satellite Imagery)")
    print("=" * 70)

    results = []
    for item in manifest:
        pair_id = item['pair_id']
        pair_dir = os.path.join(base_dir, pair_id)
        metrics = evaluate_pair(pair_dir)
        metrics['pair_id'] = pair_id
        results.append(metrics)

        print(f"\nPair: {pair_id}")
        print(f"  Ground Truth Change: {metrics['gt_change_percent']}%")
        print(f"  Predicted Change:    {metrics['pred_change_percent']}%")
        print(f"  Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1_score']} | IoU: {metrics['iou']}")
        print(f"  Regions Count:       {metrics['regions_detected']}")

    avg_f1 = np.mean([r['f1_score'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    avg_prec = np.mean([r['precision'] for r in results])
    avg_rec = np.mean([r['recall'] for r in results])

    print("\n" + "=" * 70)
    print(" AGGREGATE EVALUATION SUMMARY:")
    print(f"  Total Pairs Evaluated: {len(results)}")
    print(f"  Mean Precision:        {avg_prec:.4f}")
    print(f"  Mean Recall:           {avg_rec:.4f}")
    print(f"  Mean F1-Score:         {avg_f1:.4f}")
    print(f"  Mean IoU:              {avg_iou:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
