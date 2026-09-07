"""
ChangeChat Service — Pretrained frozen bi-temporal change detection VLM.
No fine-tuning, no adapter loaded. Pure inference on paired images.
"""
import logging
from typing import Optional

logger = logging.getLogger("satquery_ai.changechat")

try:
    import torch
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ChangeChatService:
    """Generic bi-temporal change VLM prototype — NOT official ChangeChat.
    Using microsoft/Phi-3.5-vision-instruct as placeholder (no official ChangeChat
    checkpoint available in environment). No fine-tuning performed."""

    def __init__(self, model_id: str = "microsoft/Phi-3.5-vision-instruct", device: str = "cuda"):
        self.model_id = model_id
        self.device = "cuda" if (device == "cuda" and HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self._loaded = False
        self._processor = None
        self._model = None

    def load(self) -> bool:
        if self._loaded:
            return True
        if not HAS_TRANSFORMERS or not HAS_TORCH:
            logger.warning("ChangeChat unavailable: missing torch/transformers.")
            return False
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            quant_cfg = None
            if self.device == "cuda":
                quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                quantization_config=quant_cfg,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info("ChangeChat loaded (frozen, no adapter).")
            return True
        except Exception as exc:
            logger.error("ChangeChat load failed: %s", exc)
            return False

    def describe_change(
        self, image_t1_path: str, image_t2_path: str, max_tokens: int = 256
    ) -> str:
        if not self.load():
            return "[ChangeChat] Model unavailable — frozen pretrained model load failed."
        try:
            from PIL import Image as PILImage
            img1 = PILImage.open(image_t1_path).convert("RGB")
            img2 = PILImage.open(image_t2_path).convert("RGB")
        except Exception as exc:
            return f"[ChangeChat] Image load error: {exc}"
        prompt = (
            f"<|user|>\n<|image_1|>\n<|image_2|>\n"
            f"Image 1 (before) and Image 2 (after). What changed?\n<|end|>\n<|assistant|>"
        )
        try:
            inputs = self._processor(
                text=[prompt], images=[[img1, img2]], return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            answer = self._processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            return answer or "[ChangeChat] No change detected."
        except Exception as exc:
            return f"[ChangeChat] Inference error: {exc}"


def load_changechat() -> ChangeChatService:
    return ChangeChatService()


def changechat_analyze(image_t1: str, image_t2: str, query: str = "What changed between these images?") -> str:
    """Official wrapper: frozen ChangeChat inference, no fine-tuning."""
    svc = load_changechat()
    return svc.describe_change(image_t1, image_t2)
