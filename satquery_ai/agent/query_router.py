"""
Query Router for SatQuery AI (New Architecture).

Routes natural-language queries to the correct specialist model:
  - GeoChat      → Single-image VQA, captioning, text-guided grounding
  - ChangeChat   → Bi-temporal change detection + change VQA
  - Fusion Adapter → SAR + Optical cross-modal joint analysis

Each specialist loads only when needed (lazy loading).
Output flows through Evidence Fusion for unified response.
"""
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from satquery_ai.schemas.result_schema import FusedResponse
from satquery_ai.models.remote_vlm_client import RemoteVLMClient

logger = logging.getLogger("satquery_ai.query_router")


@dataclass
class RouteDecision:
    """The result of classifying a query into a specialist path."""
    specialist: str              # "remote_vlm" | "changechat" | "fusion_adapter"
    sub_task: str                # "vqa" | "caption" | "grounding" | "change_vqa" | "joint_reasoning"
    query: str
    image_paths: List[str]
    confidence: float
    reasoning: str


@dataclass
class SpecialistOutput:
    """Raw output from a single specialist before evidence fusion."""
    specialist: str
    sub_task: str
    answer: str
    bounding_boxes: List[List[int]]
    labels: List[str]
    confidence: float
    model_trace: Dict[str, Any]
    execution_time_ms: float
    # Optional remote VLM metadata (present when using RemoteVLMClient)
    remote_metadata: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None


class QueryRouter:
    """
    Hierarchical query router.

    Flow:
      Natural-language query
           │
           ▼
    QueryRouter.classify()
           │
    ┌──────┼──────────────────┐
    │      │                  │
    ▼      ▼                  ▼
  GeoChat  ChangeChat      Fusion Adapter
  (single  (bi-temporal    (SAR + Optical
   image)   change)         fusion)
    │      │                  │
    ▼      ▼                  ▼
  Evidence Fusion layer merges into unified FusedResponse
    │
    ▼
  answer + spatial evidence + confidence + model trace
    """

    SPECIALIST_MAP = {
        "remote_vlm": {
            "label": "Remote VLM",
            "description": "Single-image VQA, captioning, and text-guided grounding (Qwen2-VL + BigEarthNet LoRA via Hugging Face Space)",
        },
        "changechat": {
            "label": "ChangeChat",
            "description": "Bi-temporal change detection and change VQA",
        },
        "fusion_adapter": {
            "label": "Fusion Adapter",
            "description": "SAR + Optical cross-modal joint analysis",
        },
    }

    def __init__(self):
        self._remote_vlm: Optional[RemoteVLMClient] = None
        self._changechat_model: Optional[Any] = None
        self._fusion_adapter_model: Optional[Any] = None
        self._load_times: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────────────────

    def classify(self, query: str, image_paths: List[str]) -> RouteDecision:
        """
        Classify query into specialist path.

        Priority:
          1. If 2+ images AND query mentions SAR/radar → Fusion Adapter
          2. If 2+ images AND query mentions change/temporal → ChangeChat
          3. Otherwise → GeoChat (single-image VQA, grounding, captioning)
        """
        query_lower = query.lower()
        num_images = len(image_paths)

        # Detect multimodal SAR + Optical pair by filename patterns or metadata
        is_multimodal_sar_optical = False
        if num_images >= 2:
            paths_lower = [str(p).lower() for p in image_paths]
            has_sar = any("s1" in p or "sar" in p or "radar" in p for p in paths_lower)
            has_optical = any("s2" in p or "optical" in p for p in paths_lower)
            # Also detect from image filenames: s1 = SAR, s2 = Optical
            filenames = [os.path.basename(p).lower() for p in image_paths]
            has_s1 = any("s1" in f for f in filenames)
            has_s2 = any("s2" in f for f in filenames)
            if (has_sar and has_optical) or (has_s1 and has_s2):
                is_multimodal_sar_optical = True

        # Priority 1: SAR-Optical fusion (multimodal pair OR 2 images + SAR keywords)
        if num_images >= 2 and (is_multimodal_sar_optical or any(
            k in query_lower for k in ["sar", "radar", "c-band", "sentinel-1", "cross-modal", "fusion"]
        )):
            return RouteDecision(
                specialist="fusion_adapter",
                sub_task="joint_reasoning",
                query=query,
                image_paths=image_paths,
                confidence=None,
                reasoning="SAR-Optical cross-modal fusion: 2 images detected (multimodal S1+S2 or SAR keywords).",
            )

        # Priority 2: Bi-temporal change (2+ images + temporal keywords)
        if num_images >= 2 and any(
            k in query_lower for k in ["change", "before", "after", "temporal", "between", "difference", "flood", "expansion", "contraction"]
        ):
            return RouteDecision(
                specialist="changechat",
                sub_task="change_vqa",
                query=query,
                image_paths=image_paths,
                confidence=None,
                reasoning="Bi-temporal change detection: 2+ images with temporal keywords.",
            )

        # Priority 3: Text-guided grounding (explicit keywords)
        if any(k in query_lower for k in ["highlight", "locate", "where is", "bounding box", "ground", "find the", "pinpoint", "detect"]):
            return RouteDecision(
                specialist="remote_vlm",
                sub_task="grounding",
                query=query,
                image_paths=image_paths,
                confidence=None,
                reasoning="Text-guided region grounding on single image (confidence from model, not hardcoded).",
            )

        # Default: Single-image VQA / captioning
        return RouteDecision(
            specialist="remote_vlm",
            sub_task="vqa",
            query=query,
            image_paths=image_paths,
            confidence=None,
            reasoning="Default single-image VQA/captioning via Remote VLM (Qwen2-VL + BigEarthNet LoRA via Hugging Face Space).",
        )

    # ──────────────────────────────────────────────────────────
    # Specialist Loading (lazy)
    # ──────────────────────────────────────────────────────────

    def _get_remote_vlm(self) -> RemoteVLMClient:
        """Return the RemoteVLMClient singleton (lazy init)."""
        if self._remote_vlm is not None:
            return self._remote_vlm

        try:
            from satquery_ai.models.remote_vlm_client import RemoteVLMClient
            self._remote_vlm = RemoteVLMClient()
            self._load_times["remote_vlm"] = time.time()
            logger.info("RemoteVLMClient (Hugging Face Space) ready.")
        except Exception as exc:
            logger.error(f"RemoteVLMClient init failed: {exc}")
            self._remote_vlm = None
        return self._remote_vlm

    # Alias for backward compatibility
    _get_geochat = _get_remote_vlm

    def _get_changechat(self) -> Any:
        """Load ChangeChat model (RemoteSensingVLM with CDVQA adapter) lazily."""
        if self._changechat_model is not None:
            return self._changechat_model

        try:
            from satquery_ai.models.remote_sensing_vlm import load_rs_vlm
            self._changechat_model = load_rs_vlm(adapter="cdvqa")
            self._load_times["changechat"] = time.time()
            logger.info("ChangeChat model loaded.")
        except Exception as exc:
            logger.error(f"ChangeChat load failed: {exc}")
            self._changechat_model = False
        return self._changechat_model

    def _get_fusion_adapter(self) -> Any:
        """Load Fusion Adapter (RemoteSensingVLM with SEN12MS SAR-Optical adapter) lazily."""
        if self._fusion_adapter_model is not None:
            return self._fusion_adapter_model

        try:
            from satquery_ai.models.remote_sensing_vlm import load_rs_vlm
            self._fusion_adapter_model = load_rs_vlm(adapter="sen12ms_sar_optical")
            self._load_times["fusion_adapter"] = time.time()
            logger.info("Fusion Adapter loaded.")
        except Exception as exc:
            logger.error(f"Fusion Adapter load failed: {exc}")
            self._fusion_adapter_model = False
        return self._fusion_adapter_model

    # ──────────────────────────────────────────────────────────
    # Specialist Execution
    # ──────────────────────────────────────────────────────────

    def _execute_remote_vlm(self, decision: RouteDecision) -> SpecialistOutput:
        """Execute Remote VLM specialist: VQA, captioning, or grounding via RemoteVLMClient."""
        start = time.time()
        client = self._get_remote_vlm()

        if client is None:
            return SpecialistOutput(
                specialist="remote_vlm", sub_task=decision.sub_task,
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": "remote_vlm_unavailable"}, execution_time_ms=0.0,
            )

        image_path = decision.image_paths[0] if decision.image_paths else ""

        try:
            if decision.sub_task == "caption":
                result = client.caption(image_path)
            elif decision.sub_task == "grounding":
                result = client.analyze(image_path, decision.query)
            else:  # vqa
                result = client.vqa(image_path, decision.query)

            elapsed = (time.time() - start) * 1000
            answer = result.get("answer", "Remote VLM returned no answer.")
            confidence = result.get("confidence")
            confidence_percent = result.get("confidence_percent")
            metadata = result.get("metadata", {})

            # Determine effective confidence for the schema
            effective_confidence = None
            if isinstance(confidence, (int, float)):
                effective_confidence = float(confidence)

            model_trace = {
                "model": metadata.get("base_model", "Qwen/Qwen2-VL-2B-Instruct"),
                "adapter": metadata.get("adapter_bucket", ""),
                "remote_service": client.space,
                "lora_verified": metadata.get("lora_verified"),
                "adapter_source": metadata.get("adapter_source"),
                "confidence_type": metadata.get("confidence_type"),
                "execution_time_ms": metadata.get("execution_time_ms", round(elapsed, 1)),
                "success": result.get("success", False),
            }

            return SpecialistOutput(
                specialist="remote_vlm", sub_task=decision.sub_task,
                answer=answer,
                bounding_boxes=[],
                labels=[decision.sub_task],
                confidence=effective_confidence,
                model_trace=model_trace,
                execution_time_ms=round(elapsed, 1),
                remote_metadata=metadata,
                raw_response=result,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return SpecialistOutput(
                specialist="remote_vlm", sub_task=decision.sub_task,
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": str(exc)}, execution_time_ms=round(elapsed, 1),
                remote_metadata={"error": str(exc)},
            )

    # Alias for backward compatibility
    _execute_geochat = _execute_remote_vlm

    def _execute_changechat(self, decision: RouteDecision) -> SpecialistOutput:
        """Execute ChangeChat specialist: bi-temporal change VQA."""
        start = time.time()
        model = self._get_changechat()

        if model is False or model is None or len(decision.image_paths) < 2:
            return SpecialistOutput(
                specialist="changechat", sub_task="change_vqa",
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": "model_not_loaded" if (model is False or model is None) else "insufficient_images"}, execution_time_ms=0.0,
            )

        try:
            answer = model.describe_change(
                decision.image_paths[0], decision.image_paths[1],
                question=decision.query,
            )
            elapsed = (time.time() - start) * 1000
            return SpecialistOutput(
                specialist="changechat", sub_task="change_vqa",
                answer=answer, bounding_boxes=[], labels=["change"],
                confidence=None,
                model_trace={"model": "RemoteSensingVLM", "adapter": "cdvqa"},
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return SpecialistOutput(
                specialist="changechat", sub_task="change_vqa",
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": str(exc)}, execution_time_ms=elapsed,
            )

    def _execute_fusion_adapter(self, decision: RouteDecision) -> SpecialistOutput:
        """Execute Fusion Adapter: SAR + Optical cross-modal analysis."""
        start = time.time()
        model = self._get_fusion_adapter()

        if model is False or model is None or len(decision.image_paths) < 2:
            return SpecialistOutput(
                specialist="fusion_adapter", sub_task="joint_reasoning",
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": "model_not_loaded" if (model is False or model is None) else "insufficient_images"}, execution_time_ms=0.0,
            )

        try:
            answer = model.describe_change(
                decision.image_paths[0], decision.image_paths[1],
                question=decision.query,
            )
            elapsed = (time.time() - start) * 1000
            return SpecialistOutput(
                specialist="fusion_adapter", sub_task="joint_reasoning",
                answer=answer, bounding_boxes=[], labels=["sar_optical"],
                confidence=None,
                model_trace={"model": "RemoteSensingVLM", "adapter": "sen12ms_sar_optical"},
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return SpecialistOutput(
                specialist="fusion_adapter", sub_task="joint_reasoning",
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": str(exc)}, execution_time_ms=elapsed,
            )

    # ──────────────────────────────────────────────────────────
    # Box parsing helper
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_boxes(text: str) -> List[List[int]]:
        """Parse bounding boxes from model output text."""
        import re
        boxes = []
        pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        for match in re.finditer(pattern, text):
            boxes.append([int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))])
        return boxes

    # ──────────────────────────────────────────────────────────
    # Main routing entry point
    # ──────────────────────────────────────────────────────────

    def route(self, query: str, image_paths: List[str]) -> Tuple[RouteDecision, SpecialistOutput]:
        """
        Full routing pipeline: classify → execute → return decision + output.
        """
        decision = self.classify(query, image_paths)
        logger.info(f"Routing: {decision.specialist}/{decision.sub_task} for query: {query[:60]}...")

        if decision.specialist == "remote_vlm":
            output = self._execute_remote_vlm(decision)
        elif decision.specialist == "changechat":
            output = self._execute_changechat(decision)
        elif decision.specialist == "fusion_adapter":
            output = self._execute_fusion_adapter(decision)
        else:
            output = SpecialistOutput(
                specialist="unknown", sub_task="unknown",
                answer="",
                bounding_boxes=[], labels=[], confidence=None,
                model_trace={"error": f"Unknown specialist: {decision.specialist}"}, execution_time_ms=0.0,
            )

        return decision, output

    def route_and_fuse(self, query: str, image_paths: List[str]) -> FusedResponse:
        """
        Route query to specialist, execute, and fuse into unified response.

        Returns FusedResponse with answer, spatial_evidence, confidence, model_trace.
        """
        overall_start = time.time()
        decision, output = self.route(query, image_paths)

        # Build spatial evidence from bounding boxes
        spatial_evidence = []
        for i, box in enumerate(output.bounding_boxes):
            spatial_evidence.append({
                "box": box,
                "label": output.labels[i] if i < len(output.labels) else "unknown",
                "confidence": output.confidence,
                "specialist": output.specialist,
            })

        # Use output's own execution time (specialist-reported) for fidelity;
        # only use overall elapsed if specialist didn't report one.
        elapsed = output.execution_time_ms or (time.time() - overall_start) * 1000

        # Build model trace
        model_trace = {
            "query": query,
            "specialist": decision.specialist,
            "sub_task": decision.sub_task,
            "model": output.model_trace.get("model", "unknown"),
            "adapter": output.model_trace.get("adapter", None),
            "execution_time_ms": round(elapsed, 1),
            "output_summary": output.answer[:200] if output.answer else "",
            # Remote VLM extras — present only for single-image VQA via RemoteVLMClient
            "remote_service": output.model_trace.get("remote_service"),
            "lora_verified": output.model_trace.get("lora_verified"),
            "adapter_source": output.model_trace.get("adapter_source"),
            "confidence_type": output.model_trace.get("confidence_type"),
        }
        # Prune None values so the trace stays clean
        model_trace = {k: v for k, v in model_trace.items() if v is not None}

        # Remote VLM metadata for display
        remote_vlm_metadata = output.remote_metadata

        return FusedResponse(
            answer=output.answer,
            spatial_evidence=spatial_evidence,
            confidence=output.confidence,
            model_trace=model_trace,
            specialist_used=decision.specialist,
            sub_task=decision.sub_task,
            execution_time_ms=round(elapsed, 1),
            remote_vlm_metadata=remote_vlm_metadata,
        )


# ─────────────────────────────────────────────────────────────────
# Evidence Fusion Layer
# ─────────────────────────────────────────────────────────────────

class EvidenceFusion:
    """
    Merges outputs from multiple specialists into unified response.

    Currently used for queries spanning multiple domains.
    Future: multi-model consensus, confidence-weighted voting.
    """

    @staticmethod
    def fuse(responses: List[FusedResponse]) -> FusedResponse:
        """Fuse multiple specialist responses into one."""
        if not responses:
            return FusedResponse(
                answer="",
                spatial_evidence=[], confidence=None,
                model_trace={"error": "no_responses_to_fuse"}, specialist_used="none", sub_task="none",
                execution_time_ms=0.0,
            )

        if len(responses) == 1:
            return responses[0]

        # Weighted confidence merge (handling None values)
        valid_confs = [r.confidence for r in responses if r.confidence is not None]
        avg_conf = round(sum(valid_confs) / len(valid_confs), 2) if valid_confs else None

        # Combine answers
        answers = [r.answer for r in responses if r.answer]
        combined_answer = " | ".join(answers[:3])  # Top 3

        # Merge spatial evidence
        all_evidence = []
        for r in responses:
            all_evidence.extend(r.spatial_evidence)

        # Merge model traces
        all_traces = [r.model_trace for r in responses]

        # Pick fastest specialist as primary
        primary = min(responses, key=lambda r: r.execution_time_ms)

        return FusedResponse(
            answer=combined_answer,
            spatial_evidence=all_evidence,
            confidence=round(avg_conf, 2),
            model_trace={
                "fused_traces": all_traces,
                "primary_specialist": primary.specialist_used,
                "fusion_method": "weighted_average",
            },
            specialist_used=f"{primary.specialist_used}+fused",
            sub_task=primary.sub_task,
            execution_time_ms=round(sum(r.execution_time_ms for r in responses), 1),
            remote_vlm_metadata=primary.remote_vlm_metadata,
        )
