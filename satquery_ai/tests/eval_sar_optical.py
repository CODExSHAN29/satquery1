"""
SAR-Optical Fusion Evaluation — SEN12MS Benchmark
Evaluates dual-stream fusion model on co-registered optical + SAR pair.
Measures logits, class probabilities, and confidence calibration.
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from satquery_ai.agent.controller import AgentController

def evaluate_sar_optical_sample(optical_path: str, sar_path: str, query: str = "Analyze land cover"):
    controller = AgentController()
    res = controller.execute_query_new(query, [optical_path, sar_path])
    trace = res.get("model_trace") or {}
    return {
        "success": res.get("success"),
        "task": res.get("task"),
        "confidence": res.get("confidence"),
        "confidence_type": trace.get("confidence_type"),
        "predicted_classes": trace.get("predicted_classes", [])[:3],
        "adapter_loaded": trace.get("checkpoint_loaded"),
        "execution_time_ms": res.get("execution_trace", {}).get("total_duration_ms"),
    }

if __name__ == "__main__":
    opt_real = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "ROIs1868_summer_s2_59_p10.png"))
    sar_real = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "ROIs1868_summer_s1_59_p10.png"))

    if os.path.exists(opt_real) and os.path.exists(sar_real):
        print(f"Testing on REAL Sentinel-1 (SAR) + Sentinel-2 (Optical) pair:\n  Optical: {os.path.basename(opt_real)}\n  SAR: {os.path.basename(sar_real)}")
        result = evaluate_sar_optical_sample(opt_real, sar_real, "Analyze land cover and fused structural features")
    else:
        import numpy as np
        from PIL import Image
        td = tempfile.mkdtemp()
        p_opt = os.path.join(td, "optical.tif")
        p_sar = os.path.join(td, "sar_s1.tif")
        Image.fromarray(np.random.randint(0,255,(256,256,3),dtype=np.uint8)).save(p_opt)
        Image.fromarray(np.random.randint(0,255,(256,256,3),dtype=np.uint8)).save(p_sar)
        print("Testing on synthetic fallback pair:")
        result = evaluate_sar_optical_sample(p_opt, p_sar)

    print("SAR-OPTICAL EVAL RESULT:", json.dumps(result, indent=2))
