import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from satquery_ai.tools.base_tool import BaseTool
from satquery_ai.utils.geotiff_parser import GeoTIFFParser


class SingleImageVQATool(BaseTool):
    """
    Single-Scene Visual Question Answering (VQA) Tool.

    Uses a fine-tuned Qwen2-VL (via BaseVLMLoader) for natural language
    question answering over satellite imagery, following RSVQA benchmark format.
    Falls back to rule-based spectral analysis when the VLM is unavailable.
    """

    def __init__(self, vlm: Optional[object] = None):
        super().__init__()
        self.vlm = vlm
        self.use_vlm = vlm is not None

    @property
    def name(self) -> str:
        return "single_image_vqa"

    @property
    def description(self) -> str:
        return "Answers natural language questions about land cover composition, feature statistics, and sensor details of a single satellite scene."

    @property
    def tool_name(self) -> str:
        return "single_image_vqa"

    @property
    def tool_description(self) -> str:
        return "Answers natural language questions about land cover composition, feature statistics, and sensor details of a single satellite scene."

    def _resolve_image(self, image_path: str) -> Image.Image:
        parser = GeoTIFFParser(image_path)
        geo_info = parser.parse(image_path)
        if isinstance(geo_info, dict) and "rgb_array" in geo_info:
            rgb_np = np.asarray(geo_info["rgb_array"], dtype=np.uint8)
        else:
            rgb_np = np.zeros((512, 512, 3), dtype=np.uint8)
        return Image.fromarray(rgb_np, mode="RGB")

    def _vqa_with_model(self, query: str, image_path: str) -> Optional[Dict[str, Any]]:
        if not self.use_vlm or self.vlm is None:
            return None
        try:
            pil_img = self._resolve_image(image_path)
            prompt = (
                f"<satellite_vqa> {query}\n"
                f"Provide a structured answer with land cover percentages "
                f"and key geographical findings."
            )
            vlm_obj = self.vlm
            answer: str = vlm_obj.generate_response(pil_img, prompt, max_new_tokens=512)  # type: ignore[attr-defined]
            # Confidence must be model-derived; the VLM does not currently expose one
            confidence = None
            return {
                "task": "SINGLE_IMAGE_VQA",
                "query": query,
                "answer": answer,
                "detected_count": 0,
                "target_label": "VQA Answer",
                "bounding_boxes": [],
                "labels": [],
                "overlay_image_path": image_path,
                "analysis": answer,
                "spectral_index_available": False,
                "spectral_index_reason": "RGB-only VLM path cannot compute NDVI/NDWI/MNDWI",
                "confidence": confidence,
            }
        except Exception as exc:
            print(f"[SingleImageVQA] VLM inference error: {exc}")
            return None

    def _vqa_with_opencv(self, query: str, image_path: str) -> Dict[str, Any]:
        parser = GeoTIFFParser(image_path)
        geo_info = parser.parse(image_path)
        if isinstance(geo_info, dict) and "rgb_array" in geo_info:
            rgb_np = np.asarray(geo_info["rgb_array"], dtype=np.uint8)
        else:
            rgb_np = np.zeros((512, 512, 3), dtype=np.uint8)

        h, w, c = rgb_np.shape
        total_pixels = h * w
        r = rgb_np[:, :, 0].astype(float)
        g = rgb_np[:, :, 1].astype(float)
        b = rgb_np[:, :, 2].astype(float)
        brightness = (r + g + b) / 3.0

        is_cloud = (brightness > 185) & (r > 165) & (g > 165) & (b > 165)
        is_veg = (g > r * 1.12) & (g > b * 1.10) & (~is_cloud)
        # Scientific note: True NDWI (McFeeters) = (GREEN - NIR) / (GREEN + NIR).
        # No NIR band is present in a 3-channel RGB image, so NDWI cannot be
        # computed. We do NOT approximate it via (g-r)/(g+r). Water detection
        # falls back to a simple brightness/NIR-proxy heuristic with no
        # spectral basis — flag it explicitly.
        spectral_index_available = False
        spectral_index_reason = "Required NIR/SWIR band unavailable"
        # Water detection: simple RGB brightness heuristic (NOT a spectral index)
        is_water = (brightness < 100) & (b > r) & (~is_cloud) & (~is_veg)
        is_urban = (brightness > 140) & (np.abs(r - g) < 25) & (~is_cloud) & (~is_water) & (~is_veg)

        water_pct = round(float((np.sum(is_water) / total_pixels) * 100), 1)
        veg_pct = round(float((np.sum(is_veg) / total_pixels) * 100), 1)
        cloud_pct = round(float((np.sum(is_cloud) / total_pixels) * 100), 1)
        urban_pct = round(float((np.sum(is_urban) / total_pixels) * 100), 1)
        barren_pct = round(max(0.0, 100.0 - (water_pct + veg_pct + cloud_pct + urban_pct)), 1)

        vqa_answer = (
            f"**Question:** {query}\n\n"
            f"**Visual Scene Analysis (RGB only — no NIR/SWIR bands):**\n"
            f"- **Water / Coastal Cover:** {water_pct}% (RGB brightness heuristic, not NDWI)\n"
            f"- **Vegetation & Forest Cover:** {veg_pct}% (RGB greenness heuristic, not NDVI)\n"
            f"- **Urban / Built-up Area:** {urban_pct}%\n"
            f"- **Atmospheric Cloud Cover:** {cloud_pct}%\n"
            f"- **Barren / Open Soil:** {barren_pct}%\n\n"
            f"**Spectral Index Note:** {spectral_index_reason}. "
            f"True NDVI/NDWI/MNDWI require NIR/SWIR bands and are not computed here.\n\n"
            f"**Key Findings:** The satellite scene primarily consists of "
            f"**{max([('Water/Coastal', water_pct), ('Vegetation', veg_pct), ('Urban', urban_pct)], key=lambda x: x[1])[0]}** "
            f"covering {w}x{h} pixels at 10m spatial resolution."
        )
        return {
            "task": "SINGLE_IMAGE_VQA",
            "query": query,
            "answer": vqa_answer,
            "detected_count": 0,
            "target_label": "VQA Scene Analysis",
            "bounding_boxes": [],
            "labels": [],
            "overlay_image_path": image_path,
            "analysis": vqa_answer,
            "spectral_index_available": spectral_index_available,
            "spectral_index_reason": spectral_index_reason,
            # Confidence is None: rule-based land-cover heuristics have no
            # model-derived uncertainty. Use the VLM path for calibrated confidence.
            "confidence": None,
        }

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        query = ""
        if len(args) > 0 and isinstance(args[0], str):
            query = args[0]
        if not query:
            query = str(kwargs.get("query", ""))

        image_path = ""
        if len(args) > 1 and isinstance(args[1], str):
            image_path = args[1]
        if not image_path:
            image_path = str(kwargs.get("image_path", ""))
        if not image_path:
            images = kwargs.get("images") or kwargs.get("image_paths") or []
            if isinstance(images, list) and len(images) > 0:
                image_path = str(images[0])
            elif isinstance(images, str):
                image_path = images

        if not image_path:
            return {"error": "No valid image path provided for SingleImageVQATool", "task": "SINGLE_IMAGE_VQA"}

        vlm_result = self._vqa_with_model(query, image_path)
        if vlm_result:
            return vlm_result
        return self._vqa_with_opencv(query, image_path)
