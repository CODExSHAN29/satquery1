"""
Remote VLM client for SatQuery AI — Hugging Face Space inference.

Heavy Qwen2-VL + BigEarthNet LoRA inference runs remotely on:
    CODEXSHAN/satquery

Environment variables:
    SATQUERY_VLM_SPACE   Space name (default: CODEXSHAN/satquery)
    HF_TOKEN             Hugging Face access token (recommended for ZeroGPU quota)
    VLM_TIMEOUT          Request timeout in seconds (default: 180)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from gradio_client import Client, handle_file

logger = logging.getLogger("satquery_ai.remote_vlm")


class RemoteVLMClient:
    """Thin wrapper around the HF Space Gradio /predict endpoint."""

    DEFAULT_SPACE = "CODEXSHAN/satquery"
    DEFAULT_TIMEOUT = 180.0
    MAX_PAYLOAD_DIM = 1024

    def __init__(
        self,
        space: Optional[str] = None,
        hf_token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.space = (
            space
            or os.getenv("SATQUERY_VLM_SPACE")
            or self.DEFAULT_SPACE
        ).strip()

        timeout_value = os.getenv("VLM_TIMEOUT")
        if timeout_value is not None:
            try:
                self.timeout = float(timeout_value)
            except ValueError:
                logger.warning(
                    "Invalid VLM_TIMEOUT=%r; falling back to %.1fs",
                    timeout_value,
                    self.DEFAULT_TIMEOUT,
                )
                self.timeout = self.DEFAULT_TIMEOUT
        elif timeout is not None:
            self.timeout = float(timeout)
        else:
            self.timeout = self.DEFAULT_TIMEOUT

        token = (
            hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or ""
        ).strip()

        self._hf_token: Optional[str] = token or None
        self._client: Optional[Client] = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    @property
    def is_authenticated(self) -> bool:
        return bool(self._hf_token)

    def _get_client(self) -> Client:
        if self._client is None:
            logger.info(
                "[RemoteVLM] Connecting to %s (authenticated=%s)",
                self.space,
                self.is_authenticated,
            )

            kwargs: Dict[str, Any] = {"verbose": False}
            if self._hf_token:
                kwargs["token"] = self._hf_token
            else:
                logger.warning(
                    "[RemoteVLM] No HF_TOKEN configured. "
                    "ZeroGPU requests may use anonymous quota."
                )

            self._client = Client(self.space, **kwargs)

        return self._client

    def _reset_client(self) -> None:
        self._client = None

    def _get_cache_key(self, image_path: str, question: str) -> str:
        try:
            stat = os.stat(image_path)
            raw = f"{stat.st_size}_{stat.st_mtime}_{question.strip().lower()}"
        except Exception:
            raw = f"{image_path}_{question.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _optimize_image_payload(
        cls, image_path: str, max_dim: int = MAX_PAYLOAD_DIM
    ) -> Tuple[str, Optional[str]]:
        """
        Compresses and optimizes high-res/GeoTIFF images before network transfer.
        Returns: (path_to_send, temporary_cleanup_path)
        """
        try:
            ext = os.path.splitext(image_path)[1].lower()
            file_size = os.path.getsize(image_path)

            # Fast bypass: standard small JPEG/PNG already under max_dim
            if ext in {".jpg", ".jpeg", ".png"} and file_size < 600 * 1024:
                try:
                    with Image.open(image_path) as im:
                        w, h = im.size
                        if max(w, h) <= max_dim:
                            return image_path, None
                except Exception:
                    pass

            # Read image array (via GeoTIFFParser for satellite TIFFs or Pillow for others)
            im: Optional[Image.Image] = None
            if ext in {".tif", ".tiff"}:
                try:
                    from satquery_ai.utils.geotiff_parser import GeoTIFFParser
                    parsed = GeoTIFFParser().parse(image_path)
                    rgb_arr = parsed.get("rgb_array")
                    if rgb_arr is not None:
                        im = Image.fromarray(rgb_arr)
                except Exception as parse_err:
                    logger.debug("[RemoteVLM] GeoTIFFParser bypass failed: %s", parse_err)

            if im is None:
                with Image.open(image_path) as src_im:
                    im = src_im.convert("RGB")

            # Scale down if larger than max_dim (preserving aspect ratio)
            w, h = im.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Write optimized JPEG to temporary file
            fd, temp_path = tempfile.mkstemp(prefix="vlm_opt_", suffix=".jpg")
            os.close(fd)
            im.save(temp_path, format="JPEG", quality=88, optimize=True)
            return temp_path, temp_path

        except Exception as exc:
            logger.debug("[RemoteVLM] Payload optimization skipped: %s", exc)
            return image_path, None

    def analyze(self, image_path: str, question: str) -> Dict[str, Any]:
        return self._predict(image_path, question)

    def vqa(self, image_path: str, question: str) -> Dict[str, Any]:
        return self._predict(image_path, question)

    def caption(
        self,
        image_path: str,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = (
            question.strip()
            if question and question.strip()
            else (
                "Describe this remote-sensing image. "
                "Identify the dominant land-cover types and major visible "
                "spatial features. Only describe features supported by the image."
            )
        )
        return self._predict(image_path, prompt)

    def _predict_raw(self, image_path: str, question: str) -> Dict[str, Any]:
        if not image_path:
            raise ValueError("image_path is required")

        image_path = os.path.abspath(image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image path is not a file: {image_path}")
        if not question or not question.strip():
            raise ValueError("question is required")

        opt_path, cleanup_path = self._optimize_image_payload(image_path)
        try:
            client = self._get_client()

            logger.info(
                "[RemoteVLM] Calling %s /predict with image=%s (payload=%s)",
                self.space,
                os.path.basename(image_path),
                os.path.basename(opt_path),
            )

            job = client.submit(
                image=handle_file(opt_path),
                question=question.strip(),
                api_name="/predict",
            )

            result: Any = job.result(timeout=self.timeout)
        finally:
            if cleanup_path and os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                except Exception:
                    pass

        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise ValueError(
                    f"Unexpected wrapped response length: {len(result)}"
                )
            result = result[0]

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Remote VLM returned non-JSON text instead of a JSON object"
                ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                f"Unexpected remote response type: {type(result).__name__}"
            )

        return result

    def _predict(self, image_path: str, question: str) -> Dict[str, Any]:
        started = time.perf_counter()

        cache_key = self._get_cache_key(image_path, question)
        if cache_key in self._cache:
            logger.info(
                "[RemoteVLM] Serving response from in-memory LRU cache for image: %s",
                os.path.basename(image_path),
            )
            cached_resp = copy.deepcopy(self._cache[cache_key])
            if "metadata" in cached_resp and isinstance(cached_resp["metadata"], dict):
                cached_resp["metadata"]["cached"] = True
                cached_resp["metadata"]["execution_time_ms"] = 0.5
            return cached_resp

        try:
            result = self._predict_raw(image_path, question)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            error_type = type(exc).__name__

            logger.exception(
                "[RemoteVLM] Prediction failed (%s): %s",
                error_type,
                exc,
            )

            self._reset_client()

            message = str(exc).strip() or "Unknown remote inference error"

            # Enhanced diagnostic for ZeroGPU quota/auth failures
            if "ZeroGPU runs limit" in message or "quota" in message.lower():
                auth_hint = ""
                if not self._hf_token:
                    auth_hint = (
                        " No HF_TOKEN configured. Set HF_TOKEN in your environment or "
                        "Streamlit secrets (must have 'Write' scope) before launching."
                    )
                else:
                    auth_hint = (
                        " HF_TOKEN is set but may be expired or invalid. "
                        "Generate a new token at https://huggingface.co/settings/tokens (Write scope)."
                    )
                message = (
                    "[VLM UNAVAILABLE] Hugging Face ZeroGPU rejected the request."
                    f"{auth_hint} Space: {self.space}. Quota info: free tier gives ~5 min/day. "
                    f"Current usage: check https://huggingface.co/spaces/{self.space}. "
                    f"Original error: {message}"
                )

            return self._error(
                message=message,
                error_type=error_type,
                execution_time_ms=elapsed_ms,
            )

        if result.get("success") is False:
            return self._error(
                message=str(
                    result.get("error")
                    or "Remote VLM returned success=false"
                ),
                error_type=str(
                    result.get("error_type")
                    or "RemoteError"
                ),
                execution_time_ms=elapsed_ms,
                raw_response=result,
            )

        answer = result.get("answer")
        if not answer:
            return self._error(
                message="Remote VLM returned no answer field",
                error_type="MissingField",
                execution_time_ms=elapsed_ms,
                raw_response=result,
            )

        confidence = result.get("confidence")
        confidence_percent = result.get("confidence_percent")
        confidence_type = result.get("confidence_type")

        if isinstance(confidence, (int, float)):
            confidence = float(confidence)
        else:
            confidence = None

        if isinstance(confidence_percent, (int, float)):
            confidence_percent = float(confidence_percent)
        elif confidence is not None:
            confidence_percent = round(confidence * 100.0, 2)
        else:
            confidence_percent = None

        remote_runtime = result.get("execution_time_ms")
        if not isinstance(remote_runtime, (int, float)):
            remote_runtime = elapsed_ms

        metadata: Dict[str, Any] = {
            "task": result.get("task"),
            "base_model": result.get("base_model"),
            "adapter_bucket": result.get("adapter_bucket"),
            "peft_type": result.get("peft_type"),
            "lora_rank": result.get("lora_rank"),
            "lora_alpha": result.get("lora_alpha"),
            "adapter_source": result.get("adapter_source"),
            "remote_sensing_adapted": result.get("remote_sensing_adapted"),
            "trained_by_us": result.get("trained_by_us"),
            "lora_verified": result.get("lora_verified"),
            "execution_time_ms": float(remote_runtime),
            "confidence_type": confidence_type,
            "remote_service": self.space,
            "authenticated": self.is_authenticated,
        }

        response_payload = {
            "success": True,
            "answer": str(answer),
            "error": None,
            "error_type": None,
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "metadata": metadata,
            "raw_response": result,
        }

        # Cache response
        self._cache[cache_key] = copy.deepcopy(response_payload)
        return response_payload

    @staticmethod
    def _error(
        message: str,
        error_type: str = "Error",
        execution_time_ms: Optional[float] = None,
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "error": message,
            "error_type": error_type,
            "execution_time_ms": execution_time_ms,
        }

        return {
            "success": False,
            "answer": None,
            "error": message,
            "error_type": error_type,
            "confidence": None,
            "confidence_percent": None,
            "metadata": metadata,
            "raw_response": raw_response,
        }


_default_client: Optional[RemoteVLMClient] = None


def get_remote_vlm_client() -> RemoteVLMClient:
    global _default_client

    if _default_client is None:
        _default_client = RemoteVLMClient()

    return _default_client
