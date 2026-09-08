"""Model registry — only fusion adapter is trained by us."""
from typing import Dict, Any

MODEL_REGISTRY: Dict[str, Any] = {
    "remote_vlm": {
        "name": "Remote VLM (Qwen2-VL-2B-Instruct + BigEarthNet LoRA)",
        "task": "single_image_vqa",
        "pretrained": True,
        "trained_by_us": False,
        "fine_tuned": True,
        "adapter": "BigEarthNet LoRA",
    },
    "changechat": {
        "name": "ChangeChat",
        "task": "bi_temporal_change",
        "pretrained": True,
        "trained_by_us": False,
        "fine_tuned": False,
        "adapter": None,
    },
    "sar_optical_fusion": {
        "name": "SAR/Optical Fusion Adapter",
        "task": "sar_optical_fusion",
        "pretrained": False,
        "trained_by_us": True,
        "fine_tuned": True,
        "adapter": "sen12ms_fusion",
        "trainable_params_millions": 0.8,
        "frozen_encoders": ["sar_encoder", "optical_encoder"],
        "loss": "BCEWithLogitsLoss",
        "dataset": "SEN12MS",
    },
}

def get_registry() -> Dict[str, Any]:
    return MODEL_REGISTRY

def is_trained_by_us(model_key: str) -> bool:
    return MODEL_REGISTRY.get(model_key, {}).get("trained_by_us", False)
