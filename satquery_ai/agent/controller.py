import logging
from typing import Dict, Any, List, Optional, Union

from satquery_ai.tools.base_tool import BaseTool
from satquery_ai.tools.text_grounding import TextGroundingTool
from satquery_ai.tools.single_image_vqa import SingleImageVQATool
from satquery_ai.tools.bitemporal_change import BiTemporalChangeTool
from satquery_ai.tools.optical_sar_joint import OpticalSARJointTool
from satquery_ai.agent.trace_logger import TraceLogger
from satquery_ai.agent.query_router import QueryRouter
from satquery_ai.schemas.result_schema import FusedResponse

logger = logging.getLogger("satquery_ai.controller")


class AgentController:
    """
    Central Agent Controller for SatQuery AI (New Architecture).

    Flow:
      Natural-language query → QueryRouter.classify() → Specialist → Evidence Fusion → Response

    Specialists:
      - GeoChat (RemoteSensingVLM): Single-image VQA, captioning, grounding
      - ChangeChat: Bi-temporal change detection + VQA
      - Fusion Adapter: SAR + Optical cross-modal analysis

    Each specialist loads lazily — only when a query matches its domain.
    Execution traces are logged for every query.
    """

    def __init__(self):
        self.trace_logger = TraceLogger()
        self.query_router = QueryRouter()

        # Legacy tools for fallback / non-VLM queries
        self.tools: Dict[str, BaseTool] = {
            "text_grounding": TextGroundingTool(),
            "single_image_vqa": SingleImageVQATool(),
            "bitemporal_change_detection": BiTemporalChangeTool(),
            "optical_sar_joint": OpticalSARJointTool(),
        }

        self._use_new_architecture = True

    # ──────────────────────────────────────────────────────
    # Query Classification (delegates to QueryRouter)
    # ──────────────────────────────────────────────────────

    def classify_query(self, query: str, image_paths: List[str]) -> Dict[str, Any]:
        """Classify query into specialist path using QueryRouter."""
        decision = self.query_router.classify(query, image_paths)
        return {
            "specialist": decision.specialist,
            "sub_task": decision.sub_task,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "image_count": len(image_paths),
        }

    # ──────────────────────────────────────────────────────
    # Main execution: New Architecture path
    # ──────────────────────────────────────────────────────

    def execute_query_new(
        self, query: str, images: Optional[List[str]] = None, task_mode: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute via new QueryRouter → Specialist → Evidence Fusion pipeline.
        Returns FusedResponse with answer, spatial_evidence, confidence, model_trace.
        """
        trace = self.trace_logger.start_trace(query=query, task_mode=task_mode)
        img_list: List[str] = []

        if isinstance(images, list):
            img_list = [str(x) for x in images]
        elif isinstance(images, str):
            img_list = [images]

        if not img_list:
            self.trace_logger.finalize_trace(status="error")
            return {"success": False, "error": "No image paths provided.", "execution_trace": trace.to_dict()}

        trace.selected_tool = "QueryRouter"
        logger.info(f"[NewArch] Routing query: '{query[:60]}...' with {len(img_list)} image(s)")

        try:
            # Route and execute via QueryRouter — ONE specialist execution path only
            fused: FusedResponse = self.query_router.route_and_fuse(query, img_list)

            # Log trace steps
            self.trace_logger.log_step(
                step_name="query_classification",
                tool_name="QueryRouter",
                parameters={"specialist": fused.specialist_used, "sub_task": fused.sub_task},
                status="completed"
            )
            self.trace_logger.log_step(
                step_name="specialist_execution",
                tool_name=fused.specialist_used,
                parameters={"query_length": len(query), "image_count": len(img_list)},
                status="completed",
                confidence=fused.confidence,
                output_summary=fused.answer[:100] if fused.answer else ""
            )
            self.trace_logger.finalize_trace(status="completed", confidence=fused.confidence)

            # Build result dict
            # Detect specialist execution errors: model_trace may carry
            # an "error" key set by the specialist fallback branches.
            specialist_error = fused.model_trace.get("error") if isinstance(fused.model_trace, dict) else None
            is_success = specialist_error is None

            result: Dict[str, Any] = {
                "success": is_success,
                "task": fused.sub_task.upper(),
                "answer": fused.answer if is_success else "",
                "error": specialist_error,
                "spatial_evidence": fused.spatial_evidence,
                "detected_count": len(fused.spatial_evidence),
                "confidence": fused.confidence if is_success else None,
                "model_trace": fused.model_trace,
                "routed_tool": fused.specialist_used,
                "execution_trace": trace.to_dict(),
                "trace_summary": trace.to_summary(),
                "architecture": "new_query_router",
                "remote_vlm_metadata": fused.remote_vlm_metadata,
            }
            return result

        except Exception as e:
            logger.error(f"[NewArch] Error: {str(e)}", exc_info=True)
            self.trace_logger.log_step(
                step_name="execution_error",
                tool_name="QueryRouter",
                parameters={"error": str(e)},
                status="error"
            )
            self.trace_logger.finalize_trace(status="error")
            return {"success": False, "error": f"QueryRouter execution failed: {str(e)}", "execution_trace": trace.to_dict()}

    # ──────────────────────────────────────────────────────
    # Main execution: Legacy path (fallback)
    # ──────────────────────────────────────────────────────

    def execute_query_legacy(
        self, query: str, images: Optional[List[str]] = None, task_mode: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Legacy execution path using old flat tool registry.
        Used when QueryRouter is disabled or fails.
        """
        trace = self.trace_logger.start_trace(query=query, task_mode=task_mode)

        img_list: List[str] = []
        if isinstance(images, list):
            img_list = [str(x) for x in images]
        elif isinstance(images, str):
            img_list = [images]

        if not img_list:
            self.trace_logger.finalize_trace(status="error")
            return {"success": False, "error": "No image paths provided.", "execution_trace": trace.to_dict()}

        primary_image = img_list[0]
        selected_tool_name = self.select_tool_for_query(query, image_count=len(img_list), task_mode=task_mode)
        tool = self.tools.get(selected_tool_name, self.tools["text_grounding"])
        trace.selected_tool = selected_tool_name

        logger.info(f"[Legacy] Routing to '{selected_tool_name}'")

        try:
            tool_kwargs: Dict[str, Any] = {
                "query": query,
                "image_path": primary_image,
                "images": img_list,
            }
            tool_kwargs.update(kwargs)

            exec_fn = getattr(tool, "execute", None) or getattr(tool, "run", None)
            result = exec_fn(**tool_kwargs) if callable(exec_fn) else {"error": f"Tool '{selected_tool_name}' not callable"}

            raw_confidence = result.get("confidence") if isinstance(result, dict) else None
            confidence: Optional[float] = float(raw_confidence) if isinstance(raw_confidence, (int, float)) else None

            self.trace_logger.log_step(
                step_name="tool_execution",
                tool_name=selected_tool_name,
                parameters={"image_path": primary_image},
                status="completed",
                confidence=confidence
            )
            self.trace_logger.finalize_trace(status="completed", confidence=confidence)

            if isinstance(result, dict):
                result["routed_tool"] = selected_tool_name
                result["execution_trace"] = trace.to_dict()
                result["trace_summary"] = trace.to_summary()
                result["architecture"] = "legacy"
                return result
            return {"task": "EXECUTION_COMPLETE", "result": str(result), "routed_tool": selected_tool_name, "execution_trace": trace.to_dict(), "trace_summary": trace.to_summary()}

        except Exception as e:
            logger.error(f"[Legacy] Error: {str(e)}", exc_info=True)
            self.trace_logger.log_step(
                step_name="tool_execution",
                tool_name=selected_tool_name,
                parameters={"image_path": primary_image},
                status="error",
                error=str(e)
            )
            self.trace_logger.finalize_trace(status="error")
            return {"error": f"Failed to process query: {str(e)}", "execution_trace": trace.to_dict()}

    # ──────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────

    def execute_query(
        self, query: str, images: Optional[List[str]] = None, task_mode: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Main execution entry point.
        Uses new QueryRouter architecture by default.
        Falls back to legacy path if QueryRouter fails.
        """
        if self._use_new_architecture:
            try:
                return self.execute_query_new(query, images, task_mode, **kwargs)
            except Exception as e:
                logger.warning(f"[Controller] New arch failed, falling back to legacy: {e}")
                return self.execute_query_legacy(query, images, task_mode, **kwargs)
        else:
            return self.execute_query_legacy(query, images, task_mode, **kwargs)

    def process_query(
        self, query: str = "", image_paths: Optional[Union[List[str], str]] = None,
        images: Optional[Union[List[str], str]] = None, task_mode: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Normalize image paths and delegate to execute_query."""
        raw_paths = image_paths or images or []
        normalized_paths: List[str] = []

        if isinstance(raw_paths, str):
            normalized_paths = [raw_paths]
        elif isinstance(raw_paths, list):
            normalized_paths = [str(p) for p in raw_paths]

        return self.execute_query(query=query, images=normalized_paths, task_mode=task_mode, **kwargs)

    # ──────────────────────────────────────────────────────
    # Legacy tool selection (for backward compatibility)
    # ──────────────────────────────────────────────────────

    def select_tool_for_query(self, query: str, image_count: int = 1, task_mode: str = "") -> str:
        """Legacy flat tool selector — kept for backward compatibility."""
        mode_lower = task_mode.lower()
        if "grounding" in mode_lower or "multi-object" in mode_lower:
            return "text_grounding"
        elif "vqa" in mode_lower or "visual question" in mode_lower or "question" in mode_lower:
            return "single_image_vqa"
        elif "change" in mode_lower or "bi-temporal" in mode_lower:
            return "bitemporal_change_detection"
        elif "sar" in mode_lower or "optical + sar" in mode_lower or "joint" in mode_lower:
            return "optical_sar_joint"

        query_lower = query.lower()
        if any(k in query_lower for k in ["highlight", "detect", "ground", "box", "locate", "find"]):
            return "text_grounding"
        elif any(k in query_lower for k in ["change", "before and after", "flood extent"]):
            return "bitemporal_change_detection"
        elif any(k in query_lower for k in ["sar", "radar", "c-band"]):
            return "optical_sar_joint"

        return "single_image_vqa"
