"""Unified result schema for ISRO-compliant output."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EvidenceItem:
    specialist: str
    label: str
    box: Optional[List[int]] = None
    confidence: float = 0.0
    source_model: str = ""


@dataclass
class FusedResponse:
    answer: str = ""
    spatial_evidence: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    specialist_used: str = ""
    sub_task: str = ""
    model_trace: Dict[str, Any] = field(default_factory=dict)
    predicted_classes: List[str] = field(default_factory=list)
    bounding_boxes: List[List[int]] = field(default_factory=list)
    execution_trace: Optional[Any] = None
    execution_time_ms: float = 0.0
    # Remote VLM metadata (adapter info, LoRA verified, etc.) — set by RemoteVLMClient
    remote_vlm_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "spatial_evidence": self.spatial_evidence,
            "confidence": self.confidence,
            "specialist_used": self.specialist_used,
            "sub_task": self.sub_task,
            "model_trace": self.model_trace,
            "predicted_classes": self.predicted_classes,
            "bounding_boxes": self.bounding_boxes,
            "execution_trace": getattr(self.execution_trace, "to_dict", lambda: {})() if self.execution_trace else {},
            "remote_vlm_metadata": self.remote_vlm_metadata,
        }
