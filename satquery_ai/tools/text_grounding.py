import os
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from satquery_ai.tools.base_tool import BaseTool


class TextGroundingTool(BaseTool):
    """
    Multi-Class Multi-Object Grounding Tool (clean architecture-compliant version).
    NO GroundingDINO, NO SAM, NO OpenCV spectral fallback.
    Returns empty evidence when no pre-trained grounding model is available,
    ensuring truthful reporting (no fabricated detections or confidence values).
    """

    def __init__(self):
        super().__init__()
        self.use_model = False

    @property
    def name(self) -> str:
        return "text_grounding"

    @property
    def description(self) -> str:
        return "Text-guided region grounding (no external models — architecture compliant)"

    @property
    def tool_name(self) -> str:
        return "text_grounding"

    @property
    def tool_description(self) -> str:
        return "Returns empty evidence; no fabricated OpenCV fallback per architecture mandate"

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        query = kwargs.get("query", "")
        if not query and len(args) > 0 and isinstance(args[0], str):
            query = args[0]
        image_path = kwargs.get("image_path", "")
        if not image_path:
            images = kwargs.get("images") or kwargs.get("image_paths") or []
            if isinstance(images, list) and len(images) > 0:
                image_path = str(images[0])
            elif isinstance(images, str) and len(args) > 1:
                image_path = str(args[1])
        return {
            "task": "MULTI_OBJECT_GROUNDING",
            "query": query,
            "detected_count": 0,
            "bounding_boxes": [],
            "labels": [],
            "overlay_image_path": None,
            "analysis": "No grounding model loaded (architecture-compliant: no OpenCV fallback, no fabricated detections). Provide a pre-trained grounding model for real detections.",
            "confidence": 0.0,
            "model": None,
            "architecture_note": "OpenCV spectral fallback removed per architecture mandate. Use a real pre-trained grounding model (e.g., GroundingDINO + SAM) for detections.",
        }
