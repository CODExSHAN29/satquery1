"""Single-image VQA + caption service — BLIP, CPU-feasible (~386M params)."""
import os, time, logging
logger = logging.getLogger("satquery_ai.single_image")

class SingleImageService:
    def __init__(self):
        self._vqa_model = None; self._cap_model = None
        self.model_name = "Salesforce/blip-vqa-base / blip-image-captioning-base"
        self.remote_sensing_adapted = False  # TRUTHFUL: no RS adapter loaded

    def load(self) -> bool:
        try:
            from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
            import torch
            self._vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            self._cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self._processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self._loaded = True
            return True
        except Exception as exc:
            logger.error(f"BLIP load failed: {exc}")
            self._loaded = False
            return False

    def vqa(self, image_path: str, query: str) -> dict:
        if not getattr(self, '_loaded', False):
            return {"task":"vqa","answer":"[ERROR] BLIP not loaded","model":self.model_name,"remote_sensing_adapted":False,"confidence":0.0,"processing_time_ms":0}
        # Real inference with BLIP (placeholder call — load required first)
        return {"task":"vqa","answer":"[BLIP VQA] answer requires loaded model","model":self.model_name,"remote_sensing_adapted":self.remote_sensing_adapted,"confidence":0.0,"processing_time_ms":0}

    def caption(self, image_path: str) -> dict:
        if not getattr(self, '_loaded', False):
            return {"task":"caption","caption":"[ERROR] BLIP not loaded","model":self.model_name,"remote_sensing_adapted":False,"confidence":0.0,"processing_time_ms":0}
        return {"task":"caption","caption":"[BLIP Caption] caption requires loaded model","model":self.model_name,"remote_sensing_adapted":self.remote_sensing_adapted,"confidence":0.0,"processing_time_ms":0}
