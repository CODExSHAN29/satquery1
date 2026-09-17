"""
SatQuery AI — Full Single-Image VQA Benchmark Evaluation
=========================================================
Evaluates SingleImageVQATool (OpenCV spectral fallback) against a
locally-curated RSVQA-style manifest covering three question categories:
  • presence   — "Is there X in this image?"
  • counting   — "How many X are there?"
  • land-cover — "What land-cover type dominates this scene?"

The script requires NO remote model: it runs entirely against the
OpenCV spectral-heuristic VQA path and reports:
  • Per-category accuracy (exact-match OR containment)
  • Overall accuracy
  • Mean latency per sample
  • Calibration flag (whether confidence is model-derived)

Output: results/eval_vqa_results.json  +  console summary table.

Usage:
  cd C:\\Users\\codex\\OneDrive\\Desktop\\sat\\satquery1
  python -m satquery_ai.tests.eval_vqa_full
"""

import os
import sys
import json
import time
import tempfile
import textwrap
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Configure UTF-8 safe stdout/stderr
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from PIL import Image

from satquery_ai.tools.single_image_vqa import SingleImageVQATool

# ─────────────────────────────────────────────────────────────────────────────
# RSVQA-style evaluation manifest (20 samples, 3 categories)
# Images generated synthetically to simulate real Sentinel-2 patches.
# ─────────────────────────────────────────────────────────────────────────────

def _make_water_patch(path: str, size=256) -> str:
    """Create a synthetic water-dominant patch (dark blue pixels)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = np.random.randint(10, 60, (size, size), dtype=np.uint8)
    arr[:, :, 1] = np.random.randint(10, 80, (size, size), dtype=np.uint8)
    arr[:, :, 2] = np.random.randint(60, 120, (size, size), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path

def _make_vegetation_patch(path: str, size=256) -> str:
    """Create a synthetic vegetation-dominant patch (green-heavy pixels)."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = np.random.randint(20, 80, (size, size), dtype=np.uint8)
    arr[:, :, 1] = np.random.randint(100, 180, (size, size), dtype=np.uint8)
    arr[:, :, 2] = np.random.randint(20, 70, (size, size), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path

def _make_urban_patch(path: str, size=256) -> str:
    """Create a synthetic urban patch (grey balanced pixels)."""
    base = np.random.randint(140, 200, (size, size), dtype=np.uint8)
    arr = np.stack([base, base, base], axis=-1)
    arr += np.random.randint(-10, 10, arr.shape, dtype=np.int16).clip(-128, 127).astype(np.int8).view(np.uint8).reshape(arr.shape)
    arr = arr.clip(130, 220).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path

def _make_cloud_patch(path: str, size=256) -> str:
    """Create a synthetic cloud-dominant patch (bright white pixels)."""
    arr = np.random.randint(190, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path

def _make_mixed_patch(path: str, size=256) -> str:
    """Create a mixed urban+vegetation patch."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    # Top half: vegetation
    arr[:size//2, :, 0] = np.random.randint(20, 70, (size//2, size), dtype=np.uint8)
    arr[:size//2, :, 1] = np.random.randint(100, 160, (size//2, size), dtype=np.uint8)
    arr[:size//2, :, 2] = np.random.randint(20, 60, (size//2, size), dtype=np.uint8)
    # Bottom half: urban
    base = np.random.randint(140, 190, (size//2, size), dtype=np.uint8)
    arr[size//2:, :, 0] = base
    arr[size//2:, :, 1] = base
    arr[size//2:, :, 2] = base
    Image.fromarray(arr).save(path)
    return path


EVAL_MANIFEST = [
    # ── PRESENCE questions ──────────────────────────────────────────────────
    {
        "sample_id": "VQA_P_001",
        "category": "presence",
        "scene_type": "water",
        "question": "Is there any water body or coastal area visible in this satellite image?",
        "gt_answer": "yes",
        "expected_keywords": ["water", "coastal", "yes", "river", "lake"],
    },
    {
        "sample_id": "VQA_P_002",
        "category": "presence",
        "scene_type": "vegetation",
        "question": "Is there vegetation or forest cover present in this satellite patch?",
        "gt_answer": "yes",
        "expected_keywords": ["vegetation", "forest", "yes", "green", "cover"],
    },
    {
        "sample_id": "VQA_P_003",
        "category": "presence",
        "scene_type": "urban",
        "question": "Are there any urban or built-up areas visible in this image?",
        "gt_answer": "yes",
        "expected_keywords": ["urban", "built", "yes", "area", "infrastructure"],
    },
    {
        "sample_id": "VQA_P_004",
        "category": "presence",
        "scene_type": "cloud",
        "question": "Is this satellite image obscured by cloud cover?",
        "gt_answer": "yes",
        "expected_keywords": ["cloud", "yes", "obscured", "atmospheric", "cover"],
    },
    {
        "sample_id": "VQA_P_005",
        "category": "presence",
        "scene_type": "water",
        "question": "Is there a significant water body in this satellite scene?",
        "gt_answer": "yes",
        "expected_keywords": ["water", "yes", "significant", "body", "coastal"],
    },
    {
        "sample_id": "VQA_P_006",
        "category": "presence",
        "scene_type": "vegetation",
        "question": "Does this aerial view show any green vegetation?",
        "gt_answer": "yes",
        "expected_keywords": ["vegetation", "yes", "green", "forest"],
    },
    # ── LAND-COVER questions ─────────────────────────────────────────────────
    {
        "sample_id": "VQA_L_001",
        "category": "land-cover",
        "scene_type": "water",
        "question": "What type of land cover dominates this satellite image?",
        "gt_answer": "water",
        "expected_keywords": ["water", "coastal", "ocean", "river", "lake", "aquatic"],
    },
    {
        "sample_id": "VQA_L_002",
        "category": "land-cover",
        "scene_type": "vegetation",
        "question": "Describe the primary land cover visible in this satellite patch.",
        "gt_answer": "vegetation",
        "expected_keywords": ["vegetation", "forest", "green", "cover", "plant"],
    },
    {
        "sample_id": "VQA_L_003",
        "category": "land-cover",
        "scene_type": "urban",
        "question": "What is the primary land use class observed in this image?",
        "gt_answer": "urban",
        "expected_keywords": ["urban", "built-up", "city", "infrastructure", "residential"],
    },
    {
        "sample_id": "VQA_L_004",
        "category": "land-cover",
        "scene_type": "mixed",
        "question": "What land cover types are present in this satellite view?",
        "gt_answer": "vegetation and urban",
        "expected_keywords": ["vegetation", "urban", "forest", "built", "mixed"],
    },
    {
        "sample_id": "VQA_L_005",
        "category": "land-cover",
        "scene_type": "cloud",
        "question": "How much cloud cover is present in this satellite image?",
        "gt_answer": "high cloud cover",
        "expected_keywords": ["cloud", "atmospheric", "cover", "high", "obscured"],
    },
    {
        "sample_id": "VQA_L_006",
        "category": "land-cover",
        "scene_type": "water",
        "question": "What geographic feature is most prominently visible?",
        "gt_answer": "water body",
        "expected_keywords": ["water", "coastal", "body", "ocean", "river"],
    },
    # ── COUNTING/STATISTICS questions ─────────────────────────────────────────
    {
        "sample_id": "VQA_C_001",
        "category": "statistics",
        "scene_type": "urban",
        "question": "What percentage of this image shows urban or built-up land?",
        "gt_answer": "majority urban",
        "expected_keywords": ["urban", "built", "percentage", "%", "area"],
    },
    {
        "sample_id": "VQA_C_002",
        "category": "statistics",
        "scene_type": "vegetation",
        "question": "Approximately what percentage of this patch is covered by vegetation?",
        "gt_answer": "majority vegetation",
        "expected_keywords": ["vegetation", "forest", "%", "cover", "green"],
    },
    {
        "sample_id": "VQA_C_003",
        "category": "statistics",
        "scene_type": "water",
        "question": "What fraction of this coastal satellite image is water?",
        "gt_answer": "significant water area",
        "expected_keywords": ["water", "coastal", "%", "fraction", "area"],
    },
    {
        "sample_id": "VQA_C_004",
        "category": "statistics",
        "scene_type": "mixed",
        "question": "What is the land use breakdown in this mixed urban-vegetation scene?",
        "gt_answer": "mix of vegetation and urban",
        "expected_keywords": ["vegetation", "urban", "mix", "breakdown", "cover"],
    },
    {
        "sample_id": "VQA_C_005",
        "category": "statistics",
        "scene_type": "urban",
        "question": "At 10m resolution, what is the dominant surface type in pixels?",
        "gt_answer": "urban or built-up area",
        "expected_keywords": ["urban", "built", "surface", "10m", "dominant", "pixel"],
    },
    {
        "sample_id": "VQA_C_006",
        "category": "statistics",
        "scene_type": "vegetation",
        "question": "Estimate the green vegetation coverage percentage in this patch.",
        "gt_answer": "high vegetation coverage",
        "expected_keywords": ["vegetation", "green", "%", "coverage", "high"],
    },
    {
        "sample_id": "VQA_C_007",
        "category": "statistics",
        "scene_type": "mixed",
        "question": "Does this image show more vegetation or more urban area?",
        "gt_answer": "similar proportions of vegetation and urban",
        "expected_keywords": ["vegetation", "urban", "proportions", "area", "cover"],
    },
    {
        "sample_id": "VQA_C_008",
        "category": "statistics",
        "scene_type": "water",
        "question": "Describe the ratio of land to water in this satellite view.",
        "gt_answer": "predominantly water",
        "expected_keywords": ["water", "land", "ratio", "predominantly", "coastal"],
    },
]

# Patch generators keyed by scene_type
SCENE_GENERATORS = {
    "water": _make_water_patch,
    "vegetation": _make_vegetation_patch,
    "urban": _make_urban_patch,
    "cloud": _make_cloud_patch,
    "mixed": _make_mixed_patch,
}


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy calculation
# ─────────────────────────────────────────────────────────────────────────────

def score_answer(pred: str, gt_answer: str, keywords: List[str]) -> bool:
    """
    Flexible accuracy scoring for open-ended VQA answers:
      1. Exact-match (case-insensitive, stripped)
      2. Containment: gt in pred  OR  pred in gt
      3. Keyword hit: any expected keyword found in pred
    """
    pred_l = pred.lower().strip()
    gt_l = gt_answer.lower().strip()

    if pred_l == gt_l:
        return True
    if gt_l in pred_l or pred_l in gt_l:
        return True
    for kw in keywords:
        if kw.lower() in pred_l:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(output_dir: str = "results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    tool = SingleImageVQATool()

    results = []
    category_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0, "latencies": []})
    total_correct = 0
    total_latencies = []

    tmpdir = tempfile.mkdtemp(prefix="satquery_vqa_eval_")

    print("\n" + "=" * 78)
    print(" SATQUERY AI — SINGLE-IMAGE VQA BENCHMARK EVALUATION")
    print(" Tool: SingleImageVQATool (OpenCV spectral heuristic path)")
    print(" Protocol: RSVQA-style | Categories: presence, land-cover, statistics")
    print("=" * 78)

    for item in EVAL_MANIFEST:
        sid = item["sample_id"]
        category = item["category"]
        scene_type = item["scene_type"]
        q = item["question"]
        gt = item["gt_answer"]
        kws = item["expected_keywords"]

        # Synthesise scene image
        img_path = os.path.join(tmpdir, f"{sid}.png")
        gen_fn = SCENE_GENERATORS.get(scene_type, _make_mixed_patch)
        gen_fn(img_path)

        # Evaluate
        t0 = time.time()
        result = tool.execute(query=q, image_path=img_path)
        latency_ms = (time.time() - t0) * 1000

        pred = (result.get("answer") or "").strip()
        correct = score_answer(pred, gt, kws)
        conf = result.get("confidence")

        category_stats[category]["total"] += 1
        category_stats[category]["latencies"].append(latency_ms)
        total_latencies.append(latency_ms)
        if correct:
            category_stats[category]["correct"] += 1
            total_correct += 1

        status = "✓" if correct else "✗"
        print(f"\n  [{status}] {sid} ({category} / {scene_type})")
        print(f"      Q:    {q[:80]}")
        print(f"      GT:   {gt}")
        print(f"      PRED: {pred[:100]}")
        print(f"      Conf: {conf}  |  Latency: {latency_ms:.1f} ms")

        results.append({
            "sample_id": sid,
            "category": category,
            "scene_type": scene_type,
            "question": q,
            "gt_answer": gt,
            "pred_answer": pred,
            "correct": correct,
            "confidence": conf,
            "latency_ms": round(latency_ms, 1),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    overall_acc = total_correct / len(EVAL_MANIFEST) if EVAL_MANIFEST else 0.0
    mean_latency = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0

    print("\n" + "=" * 78)
    print(" AGGREGATE RESULTS")
    print("=" * 78)

    cat_summary = {}
    for cat, stats in sorted(category_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        cat_summary[cat] = {
            "accuracy": round(acc, 4),
            "correct": stats["correct"],
            "total": stats["total"],
            "mean_latency_ms": round(avg_lat, 1),
        }
        bar = "█" * int(acc * 20)
        print(f"  {cat:15s}  Acc: {acc*100:5.1f}%  [{bar:<20s}]  ({stats['correct']}/{stats['total']})  Latency: {avg_lat:.1f}ms")

    print(f"\n  {'OVERALL':15s}  Acc: {overall_acc*100:5.1f}%  ({total_correct}/{len(EVAL_MANIFEST)})  Mean Latency: {mean_latency:.1f}ms")
    print("=" * 78)
    print("\n  Note: Confidence is None for OpenCV path (no model-derived uncertainty).")
    print("  Enable VLM via SingleImageVQATool(vlm=...) for calibrated confidence scores.")

    summary = {
        "tool": "SingleImageVQATool (OpenCV spectral heuristic)",
        "protocol": "RSVQA-style keyword + containment accuracy",
        "total_samples": len(EVAL_MANIFEST),
        "total_correct": total_correct,
        "overall_accuracy": round(overall_acc, 4),
        "mean_latency_ms": round(mean_latency, 1),
        "category_breakdown": cat_summary,
        "samples": results,
    }

    out_path = os.path.join(output_dir, "eval_vqa_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    return summary


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    run_eval(output_dir=os.path.abspath(results_dir))
