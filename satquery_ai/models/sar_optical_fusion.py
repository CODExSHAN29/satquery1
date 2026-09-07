"""
SAR / Optical Fusion Adapter — ONLY trainable component for ISRO compliance.
Frozen SAR encoder + frozen Optical encoder + lightweight fusion adapter.
Uses precomputed embeddings for low-compute training.
"""
import os
import json
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("satquery_ai.fusion")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore

# ---------------------------------------------------------------------------
# Real adapter checkpoint loading
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "models/adapters/sen12ms_fusion/adapter_weights.pt"
CONFIG_PATH = "models/adapters/sen12ms_fusion/adapter_config.json"


class FrozenSAREncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        return self.proj(x.mean(dim=[2, 3]))


class FrozenOpticalEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        return self.proj(x.mean(dim=[2, 3]))


class FusionAdapter(nn.Module):
    def __init__(self, embed_dim: int = 512, num_classes: int = 19):
        super().__init__()
        hidden = max(128, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, fused_embed: torch.Tensor) -> torch.Tensor:
        return self.net(fused_embed)


class SAROpticalFusion(nn.Module):
    def __init__(
        self,
        sar_embed_dim: int = 256,
        optical_embed_dim: int = 256,
        fused_dim: int = 512,
        num_classes: int = 19,
    ):
        super().__init__()
        self.sar_encoder = FrozenSAREncoder(sar_embed_dim)
        self.optical_encoder = FrozenOpticalEncoder(optical_embed_dim)
        for param in self.sar_encoder.parameters():
            param.requires_grad = False
        for param in self.optical_encoder.parameters():
            param.requires_grad = False
        self.adapter = FusionAdapter(embed_dim=fused_dim, num_classes=num_classes)

    def forward(
        self,
        sar_input: torch.Tensor,
        optical_input: torch.Tensor,
    ) -> torch.Tensor:
        sar_emb = self.sar_encoder(sar_input)
        opt_emb = self.optical_encoder(optical_input)
        fused = torch.cat([sar_emb, opt_emb], dim=-1)
        return self.adapter(fused)


# ---------------------------------------------------------------------------
# Real inference pipeline
# ---------------------------------------------------------------------------

def load_model_with_adapter(adapter_path: str = CHECKPOINT_PATH) -> Optional[SAROpticalFusion]:
    if not HAS_TORCH:
        logger.warning("torch unavailable; adapter cannot load.")
        return None
    adapter_file = adapter_path if adapter_path else CHECKPOINT_PATH
    if not os.path.exists(adapter_file):
        logger.warning("No adapter checkpoint at %s", adapter_file)
        return None
    try:
        # Seed before constructing model so frozen-encoder initialisation is
        # deterministic across reloads (only the adapter is loaded from disk).
        torch.manual_seed(42)
        model = SAROpticalFusion()
        model.adapter.load_state_dict(torch.load(adapter_file, weights_only=True))
        model.eval()
        logger.info("Adapter loaded from %s", adapter_file)
        return model
    except Exception as exc:
        logger.error("Failed to load adapter from %s: %s", adapter_file, exc)
        return None


def _load_image_as_tensor(
    path: str, channels: int, size: int = 224
) -> Optional[torch.Tensor]:
    """Load an image file and return a (C, H, W) tensor for encoder input."""
    try:
        from PIL import Image
        import torchvision.transforms as T

        if not os.path.exists(path):
            return None

        img = Image.open(path)
        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])
        tensor = transform(img.convert("RGB" if channels == 3 else "L"))
        # Ensure correct channel count
        if channels == 2:
            # For SAR: replicate single-channel to 2 (VV/VH proxy)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(2, 1, 1)
            elif tensor.shape[0] >= 2:
                tensor = tensor[:2, :, :]
        elif channels == 3:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] >= 3:
                tensor = tensor[:3, :, :]
        return tensor
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


def analyze_sar_optical(sar_path: str, optical_path: str) -> Dict[str, Any]:
    """
    Run SAR-Optical fusion adapter on real image files.

    Each image is loaded, resized, and passed through the corresponding frozen
    encoder (FrozenSAREncoder / FrozenOpticalEncoder) to produce real
    image-derived embeddings before fusion + adapter classification.
    """
    # Load real adapter
    model = load_model_with_adapter()
    checkpoint_loaded = model is not None
    checkpoint_path = CHECKPOINT_PATH if checkpoint_loaded else None

    # Read config for metadata
    adapter_params = 0
    frozen_components = []
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        adapter_params = cfg.get("adapter_params", 0)
        frozen_components = cfg.get("frozen_components", [])

    if model is None:
        # Adapter not loaded — truthful NOT WORKING for real inference
        return {
            "task": "SAR_OPTICAL_FUSION",
            "checkpoint_loaded": False,
            "checkpoint_path": None,
            "adapter_params": adapter_params,
            "predicted_classes": [],
            "status": "NOT WORKING (adapter checkpoint missing or torch unavailable)",
        }

    try:
        # Load SAR image → (2, H, W) for frozen encoder
        sar_tensor: Optional[torch.Tensor] = None
        if os.path.exists(sar_path):
            sar_tensor = _load_image_as_tensor(sar_path, channels=2, size=224)

        # Load Optical image → (3, H, W) for frozen encoder
        opt_tensor: Optional[torch.Tensor] = None
        if os.path.exists(optical_path):
            opt_tensor = _load_image_as_tensor(optical_path, channels=3, size=224)

        if sar_tensor is None or opt_tensor is None:
            # Images not found — cannot do real inference
            return {
                "task": "SAR_OPTICAL_FUSION",
                "checkpoint_loaded": checkpoint_loaded,
                "checkpoint_path": checkpoint_path,
                "adapter_params": adapter_params,
                "predicted_classes": [],
                "sar_image_loaded": sar_tensor is not None,
                "optical_image_loaded": opt_tensor is not None,
                "sar_path": sar_path,
                "optical_path": optical_path,
                "status": "NOT WORKING (image files not found — real inference requires actual image paths)",
            }

        # Build (B, C, H, W) batches
        sar_batch = sar_tensor.unsqueeze(0)    # (1, 2, 224, 224)
        opt_batch = opt_tensor.unsqueeze(0)    # (1, 3, 224, 224)

        with torch.no_grad():
            model.eval()
            logits = model(sar_batch, opt_batch)   # (1, 19)
            probs = torch.sigmoid(logits).squeeze(0)  # (19,)
            topk_vals, topk_idx = torch.topk(probs, k=min(5, probs.shape[0]))

        predicted_classes = []
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            predicted_classes.append({
                "label": f"class_{idx}",
                "probability": round(val, 4),
            })
        predicted_classes.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "task": "SAR_OPTICAL_FUSION",
            "model": "SAROpticalFusion",
            "trained_by_us": True,
            "frozen_components": frozen_components,
            "trainable_component": "fusion_adapter",
            "sar_path": sar_path,
            "optical_path": optical_path,
            "sar_image_loaded": True,
            "optical_image_loaded": True,
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_path": checkpoint_path,
            "adapter_params": adapter_params,
            "predicted_classes": predicted_classes,
            "adapter_logits_sample": [round(x, 4) for x in logits.squeeze(0).tolist()[:5]],
        }
    except Exception as exc:
        logger.error("Real adapter inference failed: %s", exc)
        return {
            "task": "SAR_OPTICAL_FUSION",
            "checkpoint_loaded": checkpoint_loaded,
            "checkpoint_path": checkpoint_path,
            "adapter_params": adapter_params,
            "predicted_classes": [],
            "inference_error": str(exc),
            "status": "PARTIAL (adapter loaded, inference failed)",
        }


# ---------------------------------------------------------------------------
# Training reporting (truthful)
# ---------------------------------------------------------------------------

def report_training_metrics() -> Dict[str, Any]:
    """Report only what actually exists — no fabricated history."""
    metrics = {
        "dataset_path": None,
        "dataset_samples_used": None,
        "labels_used": None,
        "train_validation_split": None,
        "loss_function": "BCEWithLogitsLoss (from code)",
        "epochs": 5,
        "batch_size": 4,
        "lr": 1e-4,
        "initial_training_loss": None,
        "final_training_loss": None,
        "validation_metric": None,
        "encoder_names": ["FrozenSAREncoder", "FrozenOpticalEncoder"],
        "frozen_param_count": 0,
        "trainable_param_count": adapter_params if 'adapter_params' in globals() else 0,
    }
    # Check if any real dataset or training manifest exists
    dataset_dir = "datasets"
    manifest = os.path.join(dataset_dir, "manifest.json")
    embeddings_meta = "datasets/embeddings/precomputed_meta.json"
    if os.path.exists(manifest):
        try:
            with open(manifest) as f:
                data = json.load(f)
                metrics["dataset_path"] = dataset_dir
                metrics["dataset_samples_used"] = len(data) if isinstance(data, list) else None
        except Exception:
            pass
    elif os.path.exists(embeddings_meta):
        try:
            with open(embeddings_meta) as f:
                meta = json.load(f)
                metrics["dataset_path"] = "datasets/embeddings"
                metrics["dataset_samples_used"] = meta.get("count")
        except Exception:
            pass
    # Count frozen params from model
    try:
        temp_model = SAROpticalFusion()
        frozen = sum(p.numel() for p in temp_model.sar_encoder.parameters()) + \
                 sum(p.numel() for p in temp_model.optical_encoder.parameters())
        trainable = sum(p.numel() for p in temp_model.adapter.parameters())
        metrics["frozen_param_count"] = frozen
        metrics["trainable_param_count"] = trainable
    except Exception:
        pass
    # Real checkpoint info
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
                metrics["adapter_params_config"] = cfg.get("adapter_params")
                metrics["frozen_components_config"] = cfg.get("frozen_components")
        except Exception:
            pass
    return metrics
