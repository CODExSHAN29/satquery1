"""
Mocked tests for RemoteVLMClient, routing, and pipeline integration.
No live internet dependency.
"""
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from satquery_ai.models.remote_vlm_client import RemoteVLMClient
from satquery_ai.agent.query_router import QueryRouter, RouteDecision
from satquery_ai.schemas.result_schema import FusedResponse


class TestRemoteVLMClientMocked(unittest.TestCase):
    def test_client_init_uses_env_defaults(self):
        with patch.dict(os.environ, {"SATQUERY_VLM_SPACE": "TEST/space", "HF_TOKEN": "tok", "VLM_TIMEOUT": "30"}):
            client = RemoteVLMClient()
            self.assertEqual(client.space, "TEST/space")
            self.assertEqual(client._hf_token, "tok")
            self.assertEqual(client.timeout, 30.0)

    def test_predict_success_dict(self):
        client = RemoteVLMClient()
        mock_result = {
            "success": True,
            "answer": "Flooding visible.",
            "confidence": 0.92,
            "confidence_percent": 92.0,
            "task": "vqa",
            "base_model": "Qwen/Qwen2-VL-2B-Instruct",
            "adapter_bucket": "CODEXSHAN/satquery-qwen2vl-bigearthnet",
            "lora_verified": True,
            "adapter_source": "huggingface",
            "execution_time_ms": 420.5,
            "confidence_type": "generation",
        }
        with patch.object(client, "_predict_raw", return_value=mock_result):
            result = client.vqa("fake_path.tif", "What changed?")

        self.assertTrue(result["success"])
        self.assertEqual(result["answer"], "Flooding visible.")
        self.assertEqual(result["confidence"], 0.92)
        self.assertEqual(result["confidence_percent"], 92.0)
        self.assertIsNotNone(result["metadata"].get("lora_verified"))
        self.assertTrue(result["metadata"].get("lora_verified"))

    def test_predict_null_confidence_shows_unavailable(self):
        client = RemoteVLMClient()
        mock_result = {
            "success": True,
            "answer": "No objects.",
            "confidence": None,
            "confidence_percent": None,
            "metadata": {"task": "caption"},
        }
        with patch.object(client, "_predict_raw", return_value=mock_result):
            result = client.caption("fake.tif")

        self.assertTrue(result["success"])
        self.assertIsNone(result.get("confidence"))
        self.assertIsNone(result.get("confidence_percent"))

    def test_predict_error_response(self):
        client = RemoteVLMClient()
        mock_result = {"success": False, "error": "Space sleeping"}
        with patch.object(client, "_predict_raw", return_value=mock_result):
            result = client.vqa("fake.tif", "Q")

        self.assertFalse(result["success"])
        self.assertIn("Space sleeping", result["error"])
        self.assertEqual(result["metadata"]["error_type"], "RemoteError")

    def test_predict_exception_captured(self):
        client = RemoteVLMClient()
        with patch.object(client, "_predict_raw", side_effect=Exception("timeout")):
            result = client.vqa("fake.tif", "Q")
        self.assertFalse(result["success"])
        self.assertIn("timeout", result["metadata"]["error"])
        self.assertEqual(result["metadata"]["error_type"], "Exception")

    def test_predict_raw_file_not_found(self):
        client = RemoteVLMClient()
        with self.assertRaises(FileNotFoundError):
            client._predict_raw("non_existent_file_xyz_123.jpg", "question")


class TestQueryRouterRemoteIntegration(unittest.TestCase):
    def test_route_vqa_calls_remote_vlm_once(self):
        router = QueryRouter()
        with patch.object(router, "_execute_remote_vlm") as mock_vlm, \
             patch.object(router, "_execute_changechat") as mock_change, \
             patch.object(router, "_execute_fusion_adapter") as mock_fusion:
            mock_vlm.return_value = MagicMock(
                answer="Urban.", confidence=0.8, bounding_boxes=[], labels=["vqa"],
                model_trace={"model": "Qwen"}, execution_time_ms=150.0,
                remote_metadata={"task": "vqa"},
            )
            router.route("What is here?", ["img.tif"])
            self.assertEqual(mock_vlm.call_count, 1)
            self.assertEqual(mock_change.call_count + mock_fusion.call_count, 0)

    def test_route_fusion_adapter_for_s1_s2(self):
        router = QueryRouter()
        decision = router.classify("Analyze SAR and optical.", ["S1.tif", "S2.tif"])
        self.assertEqual(decision.specialist, "fusion_adapter")

    def test_fused_response_has_remote_vlm_metadata(self):
        router = QueryRouter()
        with patch.object(router, "_execute_remote_vlm") as mock_vlm:
            mock_vlm.return_value = MagicMock(
                answer="Water bodies.", bounding_boxes=[[10, 10, 20, 20]],
                labels=["vqa"], confidence=0.85, model_trace={"model": "Qwen"},
                execution_time_ms=120.0, remote_metadata={"lora_verified": True},
            )
            fused = router.route_and_fuse("Highlight water.", ["img.tif"])
        self.assertIsNotNone(fused.remote_vlm_metadata)
        self.assertTrue(fused.remote_vlm_metadata.get("lora_verified"))


class TestFusedResponseSchema(unittest.TestCase):
    def test_to_dict_includes_remote_vlm_metadata(self):
        resp = FusedResponse(
            answer="Test",
            remote_vlm_metadata={"adapter_bucket": "bucket"},
        )
        d = resp.to_dict()
        self.assertIn("remote_vlm_metadata", d)
        self.assertEqual(d["remote_vlm_metadata"]["adapter_bucket"], "bucket")


class TestAgentControllerAndRouterIntegration(unittest.TestCase):
    def test_controller_success_response_shape(self):
        from satquery_ai.agent.controller import AgentController

        controller = AgentController()
        mock_fused = FusedResponse(
            answer="Agricultural area.",
            spatial_evidence=[],
            confidence=0.88,
            model_trace={"model": "Qwen2-VL"},
            specialist_used="remote_vlm",
            sub_task="vqa",
            execution_time_ms=250.0,
            remote_vlm_metadata={"adapter_bucket": "CODEXSHAN/satquery-qwen2vl-bigearthnet"},
        )

        with patch.object(controller.query_router, "route_and_fuse", return_value=mock_fused):
            res = controller.execute_query_new("What is here?", ["img.tif"])

        self.assertTrue(res["success"])
        self.assertIsNone(res["error"])
        self.assertEqual(res["answer"], "Agricultural area.")
        self.assertEqual(res["confidence"], 0.88)
        self.assertEqual(res["routed_tool"], "remote_vlm")
        self.assertIsNotNone(res["remote_vlm_metadata"])

    def test_controller_error_response_shape(self):
        from satquery_ai.agent.controller import AgentController

        controller = AgentController()
        mock_fused = FusedResponse(
            answer="",
            spatial_evidence=[],
            confidence=None,
            model_trace={"error": "remote_vlm_unavailable"},
            specialist_used="remote_vlm",
            sub_task="vqa",
            execution_time_ms=0.0,
            remote_vlm_metadata={"error": "remote_vlm_unavailable"},
        )

        with patch.object(controller.query_router, "route_and_fuse", return_value=mock_fused):
            res = controller.execute_query_new("What is here?", ["img.tif"])

        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "remote_vlm_unavailable")
        self.assertEqual(res["answer"], "")
        self.assertIsNone(res["confidence"])

    def test_controller_missing_images_error(self):
        from satquery_ai.agent.controller import AgentController

        controller = AgentController()
        res = controller.execute_query_new("What is here?", [])
        self.assertFalse(res["success"])
        self.assertIn("No image paths provided", res["error"])

    def test_controller_bitemporal_change_execution(self):
        from satquery_ai.agent.controller import AgentController
        import numpy as np, tempfile, os
        from PIL import Image

        # Create two synthetic test images (before/after)
        img_a = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        img_b = img_a.copy()
        img_b[64:192, 64:192] = 200  # introduce a visible change patch

        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "before.tif")
            p2 = os.path.join(td, "after.tif")
            Image.fromarray(img_a).save(p1)
            Image.fromarray(img_b).save(p2)

            controller = AgentController()
            res = controller.execute_query_new(
                "What changed between these images?",
                [p1, p2],
            )
            self.assertTrue(res["success"], f"Controller failed: {res.get('error')}")
            self.assertIn("CHANGE_VQA", res["task"])
            self.assertIsInstance(res["spatial_evidence"], list)
            self.assertIn("overlay_image_path", res)

    def test_controller_sar_optical_fusion_execution(self):
        from satquery_ai.agent.controller import AgentController
        import numpy as np, tempfile, os
        from PIL import Image

        # Create synthetic optical and SAR images
        optical = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        sar = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            p_opt = os.path.join(td, "optical.tif")
            p_sar = os.path.join(td, "sar_s1.tif")
            Image.fromarray(optical).save(p_opt)
            Image.fromarray(sar).save(p_sar)

            controller = AgentController()
            res = controller.execute_query_new(
                "Fuse SAR and optical imagery for urban analysis",
                [p_opt, p_sar],
            )
            self.assertTrue(res["success"], f"Controller failed: {res.get('error')}")
            self.assertIn("JOINT_REASONING", res["task"])
            self.assertIsInstance(res["spatial_evidence"], list)
            self.assertIn("overlay_image_path", res)

    def test_model_registry_remote_vlm_canonical(self):
        from satquery_ai.models.registry import get_registry

        registry = get_registry()
        self.assertIn("remote_vlm", registry)
        self.assertTrue(registry["remote_vlm"]["fine_tuned"])
        self.assertEqual(registry["remote_vlm"]["adapter"], "BigEarthNet LoRA")
        self.assertIn("changechat", registry)
        self.assertIn("sar_optical_fusion", registry)


class TestRemoteVLMCachingAndOptimization(unittest.TestCase):
    def test_optimize_image_payload_resizes_oversized(self):
        import numpy as np, tempfile, os
        from PIL import Image

        # Create large test image (2048 x 2048)
        large_arr = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "large.png")
            Image.fromarray(large_arr).save(src_path)

            opt_path, cleanup_path = RemoteVLMClient._optimize_image_payload(src_path, max_dim=1024)
            self.assertIsNotNone(cleanup_path)
            self.assertTrue(os.path.exists(opt_path))

            with Image.open(opt_path) as im:
                w, h = im.size
                self.assertLessEqual(max(w, h), 1024)

            if cleanup_path and os.path.exists(cleanup_path):
                os.remove(cleanup_path)

    def test_in_memory_lru_caching(self):
        import numpy as np, tempfile, os
        from PIL import Image

        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "cached_test.png")
            Image.fromarray(arr).save(src_path)

            client = RemoteVLMClient()
            mock_result = {
                "success": True,
                "answer": "Cached answer.",
                "confidence": 0.95,
                "confidence_percent": 95.0,
                "task": "vqa",
                "execution_time_ms": 500.0,
            }

            with patch.object(client, "_predict_raw", return_value=mock_result) as mock_raw:
                # First call should invoke _predict_raw
                res1 = client.vqa(src_path, "Describe scene.")
                self.assertEqual(mock_raw.call_count, 1)
                self.assertEqual(res1["answer"], "Cached answer.")

                # Second identical call should hit cache without calling _predict_raw again
                res2 = client.vqa(src_path, "Describe scene.")
                self.assertEqual(mock_raw.call_count, 1)
                self.assertEqual(res2["answer"], "Cached answer.")
                self.assertTrue(res2["metadata"].get("cached", False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
