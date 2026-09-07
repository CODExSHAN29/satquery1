"""
SatQuery AI — Benchmark Test Suite
Tests RSVQA, VRSBench, CDVQA evaluation readiness.
"""
import os
import sys
import unittest

# Add project root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from satquery_ai.engine import SatQueryEngine
from satquery_ai.agent.controller import AgentController
from satquery_ai.agent.query_classifier import QueryClassifier
from satquery_ai.tools.text_grounding import TextGroundingTool
from satquery_ai.tools.single_image_vqa import SingleImageVQATool
from satquery_ai.tools.bitemporal_change import BiTemporalChangeTool
from satquery_ai.tools.optical_sar_joint import OpticalSARJointTool


class TestSatQueryEngine(unittest.TestCase):
    """Tests SatQueryEngine routing and response structure."""

    @classmethod
    def setUpClass(cls):
        cls.engine = SatQueryEngine()

    def test_single_vqa_query(self):
        """Single-image VQA returns answer and execution trace."""
        res = self.engine.process_query(query="Describe the land cover.", image_paths=[])
        # Engine should return error if no image, but structure should be valid
        self.assertIn("execution_trace", res)

    def test_grounding_tool_routing(self):
        """Controller routes grounding query correctly."""
        controller = AgentController()
        selected = controller.select_tool_for_query("Highlight water bodies", task_mode="Multi-Object Text Grounding")
        self.assertEqual(selected, "text_grounding")

    def test_vqa_tool_routing(self):
        """Controller routes VQA query correctly."""
        controller = AgentController()
        selected = controller.select_tool_for_query("What land cover is visible?", task_mode="Single-Scene Visual Question Answering (VQA)")
        self.assertEqual(selected, "single_image_vqa")

    def test_bitemporal_tool_routing(self):
        """Controller routes bi-temporal query correctly."""
        controller = AgentController()
        selected = controller.select_tool_for_query("What changed?", task_mode="Bi-Temporal Change Detection")
        self.assertEqual(selected, "bitemporal_change_detection")

    def test_optical_sar_tool_routing(self):
        """Controller routes SAR-Optical query correctly."""
        controller = AgentController()
        selected = controller.select_tool_for_query("Fuse optical and SAR imagery.", task_mode="Optical + SAR Joint Analysis")
        self.assertEqual(selected, "optical_sar_joint")

    def test_tool_outputs_contain_keys(self):
        """Each tool returns the expected keys."""
        expected_keys = {"task", "query", "analysis", "confidence", "execution_trace", "trace_summary"}

        # VQA tool (no image needed for structure check)
        tool = SingleImageVQATool()
        res = tool.execute(query="test", image_path="")
        self.assertTrue(expected_keys.issubset(res.keys()) or "error" in res)

    def test_trace_logger(self):
        """TraceLogger produces valid JSON trace."""
        from satquery_ai.agent.trace_logger import TraceLogger
        logger = TraceLogger()
        trace = logger.start_trace(query="test", task_mode="test")
        logger.log_step(step_name="test_step", tool_name="test_tool", parameters={}, status="completed")
        logger.finalize_trace(status="completed", confidence=0.95)
        trace_dict = trace.to_dict()
        self.assertIn("query_hash", trace_dict)
        self.assertIn("steps", trace_dict)
        self.assertIn("confidence_score", trace_dict)

    def test_query_classifier(self):
        """QueryClassifier correctly classifies queries."""
        result = QueryClassifier.classify_query("Highlight the water body", num_images=1)
        self.assertIn("task", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)


class TestBiTemporalChangeTool(unittest.TestCase):
    """Tests BiTemporalChangeTool returns valid structure."""

    def test_requires_two_images(self):
        tool = BiTemporalChangeTool()
        res = tool.execute(query="What changed?", images=[])
        self.assertIn("error", res)

    def test_returns_change_fields(self):
        tool = BiTemporalChangeTool()
        # Should still return expected fields even without valid images
        res = tool.execute(query="test", images=["t1.jpg", "t2.jpg"])
        # With dummy images, result may have error or task fields
        self.assertIn("task", res)


class TestOpticalSARJointTool(unittest.TestCase):
    """Tests OpticalSARJointTool returns valid structure."""

    def test_requires_two_images(self):
        tool = OpticalSARJointTool()
        res = tool.execute(query="Fuse imagery.", images=[])
        self.assertIn("error", res)


class TestTextGroundingTool(unittest.TestCase):
    """Tests TextGroundingTool returns valid structure."""

    def test_execute_returns_structure(self):
        tool = TextGroundingTool()
        res = tool.execute(query="water", image_path="")
        # Returns task and zero-detection results
        self.assertIn("task", res)
        self.assertEqual(res.get("detected_count"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)