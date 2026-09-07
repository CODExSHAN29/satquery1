from typing import Dict, Any


class QueryClassifier:
    """
    Classifies natural language queries into target remote sensing tasks.
    Returns a dict with a single canonical 'task' key whose value matches
    the tool name keys in AgentController.tools:
      - text_grounding
      - single_image_vqa
      - bitemporal_change_detection
      - optical_sar_joint
    """

    # Mapping from classifier task names to controller tool names
    _TOOL_MAP = {
        "GROUNDING": "text_grounding",
        "OPTICAL_SAR": "optical_sar_joint",
        "BI_TEMPORAL": "bitemporal_change_detection",
        "SINGLE_VQA": "single_image_vqa",
        "VQA": "single_image_vqa",
        "CHANGE": "bitemporal_change_detection",
    }

    @staticmethod
    def classify_query(query: str, num_images: int = 1) -> Dict[str, Any]:
        q_lower = query.lower()

        # Priority 1: Optical-SAR cross-modal fusion (2 images + SAR/radar keywords)
        if num_images == 2 and any(k in q_lower for k in ["sar", "radar", "cross-modal", "fusion", "sentinel1"]):
            task = "OPTICAL_SAR"
            tool = "optical_sar_joint"
        # Priority 2: Explicit Grounding Keywords
        elif any(k in q_lower for k in ["highlight", "locate", "where is", "bounding box", "ground", "find the", "pinpoint"]) and not any(k in q_lower for k in ["change", "before", "after", "difference"]):
            task = "GROUNDING"
            tool = "text_grounding"
        # Priority 3: Bi-Temporal change detection
        elif num_images >= 2 or any(k in q_lower for k in ["change", "before", "after", "temporal", "between", "difference", "expansion"]):
            task = "BI_TEMPORAL"
            tool = "bitemporal_change_detection"
        # Priority 4: Default Single-Image VQA
        else:
            task = "SINGLE_VQA"
            tool = "single_image_vqa"

        return {
            "task": task,
            "tool": tool,          # canonical controller tool name
            "confidence": 0.85 if task == "SINGLE_VQA" else 0.92,
            "reasoning": f"Classified as {task}.",
        }