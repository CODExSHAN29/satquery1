import torch
from typing import Optional, Any
from PIL import Image

try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from peft import PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class BaseVLMLoader:
    """
    Handles loading and inference for vision-language models (Qwen2-VL / Florence-2)
    with dynamic LoRA adapter switching (BigEarthNet, RSVQA, CDVQA).
    """

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.active_adapter: Optional[str] = None

    def _safe_dtype(self) -> "torch.dtype":
        """Return bfloat16 only when CUDA supports it; safe float32 fallback otherwise."""
        if not torch.cuda.is_available():
            return torch.float32
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float32

    def load_base_model(self):
        """Loads the base VLM weights into GPU/CPU memory."""
        if not HAS_TRANSFORMERS:
            print("[Warning] `transformers` or `peft` not installed. Running in mock/simulation mode.")
            return

        # Free any previously loaded model before loading a new one
        self.unload_model()

        print(f"[SatQuery AI] Loading base VLM model: {self.model_id} on device: {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        dtype = self._safe_dtype()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        print("[SatQuery AI] Base VLM loaded successfully.")

    def unload_model(self):
        """Explicitly free GPU memory held by the current model."""
        if self.model is not None:
            del self.model
            self.model = None
            self.active_adapter = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_adapter(self, adapter_path: str, adapter_name: str):
        """Attaches a fine-tuned LoRA adapter (e.g. BigEarthNet, RSVQA, CDVQA).

        Replaces the previous adapter if one was already active.
        """
        if self.model is None or not HAS_TRANSFORMERS:
            return

        # Free previous PeftModel wrapper before loading a new one
        prev_model = self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[SatQuery AI] Attaching LoRA adapter '{adapter_name}' from {adapter_path}...")
        self.model = PeftModel.from_pretrained(prev_model, adapter_path, adapter_name=adapter_name)
        del prev_model
        self.active_adapter = adapter_name

    def generate_response(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        """Runs vision-language inference on an image + query prompt."""
        if self.model is None or self.processor is None:
            return f"[Simulation Response] Processed prompt '{prompt}' on image ({image.width}x{image.height})."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        return output_text[0]