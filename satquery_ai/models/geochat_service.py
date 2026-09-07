"""
GeoChat Service — Pretrained Frozen VLM for single-image remote-sensing tasks.
No fine-tuning, no adapters loaded by default. Pure inference.
Uses microsoft/Phi-3.5-vision-instruct.
"""
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("satquery_ai.geochat")

try:
    import torch
    from PIL import Image
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    HAS_TRANSFORMERS = False


class GeoChatService:
    """Lazy-loaded frozen GeoChat for VQA / caption / text-grounding."""

    def __init__(
        self,
        model_id: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = "cuda" if (device == "cuda" and HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self._loaded = False
        self._processor = None
        self._model = None

    def load(self) -> bool:
        if self._loaded:
            return True
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            logger.warning("GeoChat unavailable: missing torch/transformers.")
            return False
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            quant_cfg = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                if self.device == "cuda"
                else None
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                quantization_config=quant_cfg,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info("GeoChat loaded (frozen, no adapter).")
            return True
        except Exception as exc:
            logger.error("GeoChat load failed: %s", exc)
            return False

    def vqa(self, image_path: str, question: str, max_tokens: int = 256) -> str:
        if not self.load():
            return "[GeoChat] Model unavailable — frozen pretrained model load failed."
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert("RGB")
        except Exception as exc:
            return f"[GeoChat] Image load error: {exc}"
        prompt = (
            f"<|user|>\n<|image|>\n"
            f"Answer concisely: {question}\n<|end|>\n<|assistant|>"
        )
        try:
            inputs = self._processor(
                text=[prompt], images=[img], return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            answer = self._processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            return answer or "[GeoChat] No answer generated."
        except Exception as exc:
            return f"[GeoChat] Inference error: {exc}"

    def caption(self, image_path: str, max_tokens: int = 128) -> str:
        return self.vqa(
            image_path,
            "Describe land cover and geographic features in this satellite image.",
            max_tokens=max_tokens,
        )

    def ground(self, image_path: str, text_query: str, max_tokens: int = 64) -> str:
        return self.vqa(
            image_path,
            f"Find regions matching '{text_query}'. Return bounding box and label.",
            max_tokens=max_tokens,
        )


# Singleton lazy loader
def load_geochat() -> GeoChatService:
    return GeoChatService()
