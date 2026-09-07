"""
SatQuery AI - Master Engine Interface.
Unified entry point for REST API, Streamlit, and Python clients.
"""

from typing import List, Dict, Any, Optional
from satquery_ai.agent.controller import AgentController


class SatQueryEngine:
    """
    Master Core Engine interface for SatQuery AI.
    Exposes process_query() for REST API microservices, Streamlit, and Python clients.
    """

    def __init__(self, use_vlm: bool = False, model_id: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.controller = AgentController()
        self.use_vlm = use_vlm
        self.model_id = model_id
        self._vlm = None

        # Lazy-load VLM if requested
        if use_vlm:
            try:
                from satquery_ai.models.base_vlm import BaseVLMLoader
                self._vlm = BaseVLMLoader(model_id=model_id)
                self._vlm.load_base_model()
                # Wire the VLM into VQA tool
                from satquery_ai.tools.single_image_vqa import SingleImageVQATool
                vqa_tool = self.controller.tools.get("single_image_vqa")
                if isinstance(vqa_tool, SingleImageVQATool) and self._vlm is not None:
                    vqa_tool.vlm = self._vlm
                    vqa_tool.use_vlm = True
            except Exception as exc:
                print(f"[SatQueryEngine] VLM unavailable, falling back to rule-based: {exc}")
                self._vlm = None

    def process_query(self, query: str, image_paths: List[str]) -> Dict[str, Any]:
        """
        Process a user natural language query with one or more satellite images.
        """
        return self.controller.execute_query(query=query, image_paths=image_paths)

    def run(self, query: str, image_paths: List[str]) -> Dict[str, Any]:
        """Alias for process_query."""
        return self.process_query(query=query, image_paths=image_paths)
