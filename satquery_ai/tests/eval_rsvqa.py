"""
RSVQA Benchmark Evaluation — Single-Image VQA
Evaluates Remote VLM (BigEarthNet LoRA) against RSVQA test split.
Expected format per ISRO SIH26167: image + question + answer + category.
"""
import os, sys, json, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from satquery_ai.agent.controller import AgentController
from satquery_ai.models.remote_vlm_client import RemoteVLMClient

def evaluate_rsvqa_sample(image_path: str, question: str, gt_answer: str, category: str = "counting"):
    controller = AgentController()
    res = controller.execute_query_new(question, [image_path])
    pred = res.get("answer") or ""
    conf = res.get("confidence")
    conf_type = (res.get("model_trace") or {}).get("confidence_type")
    # Simple accuracy for demo (exact-match or containment)
    correct = gt_answer.lower() in pred.lower() or pred.lower() in gt_answer.lower()
    return {"category": category, "pred": pred, "gt": gt_answer, "correct": correct,
            "confidence": conf, "confidence_type": conf_type}

if __name__ == "__main__":
    img_real = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "River_28.jpg"))
    if not os.path.exists(img_real):
        img_real = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "ROIs1868_summer_s2_59_p10.png"))

    if os.path.exists(img_real):
        print(f"Testing on REAL Satellite Image: {os.path.basename(img_real)}")
        result = evaluate_rsvqa_sample(img_real, "What type of geographic or land cover feature is present in this image?", "river", "presence")
    else:
        import numpy as np
        from PIL import Image
        td = tempfile.mkdtemp()
        img_path = os.path.join(td, "rsvqa_test.png")
        Image.fromarray(np.random.randint(0,255,(512,512,3),dtype=np.uint8)).save(img_path)
        print("Testing on synthetic fallback image:")
        result = evaluate_rsvqa_sample(img_path, "How many buildings?", "3", "counting")

    print("RSVQA EVAL RESULT:", json.dumps(result, indent=2))
