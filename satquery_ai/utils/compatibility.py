"""
Compatibility layer for SatQuery AI.
Provides backward-compatible wrappers for GeoTIFF parsing and metadata extraction across tools.
"""

from typing import Any, Dict, Optional, Union, cast
from satquery_ai.utils.geotiff_parser import GeoTIFFParser


def extract_geotiff_info(file_path: Optional[str] = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compatibility wrapper for GeoTIFF metadata & band extraction.
    Safely handles classmethod or instance calls on GeoTIFFParser.
    """
    path = file_path or str(kwargs.get("image_path", "")) or str(kwargs.get("file_path", ""))

    # Safe call if extract_metadata exists on GeoTIFFParser
    extract_fn = getattr(GeoTIFFParser, "extract_metadata", None)
    if callable(extract_fn):
        try:
            res = extract_fn(path)
            if isinstance(res, dict):
                return cast(Dict[str, Any], res)
        except Exception:
            pass

    # Fallback to instantiating GeoTIFFParser
    parser = GeoTIFFParser(path)
    parse_fn = getattr(parser, "read_rgb", None) or getattr(parser, "parse", None) or getattr(parser, "get_rgb_image", None)
    if callable(parse_fn):
        res = parse_fn(path)
        if isinstance(res, dict):
            return cast(Dict[str, Any], res)

    return {
        "rgb_array": None,
        "width": 512,
        "height": 512,
        "count": 3,
        "crs": "EPSG:4326",
        "transform": None
    }


def parse_geotiff(image_path: Optional[str] = None, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Legacy wrapper function for tool compatibility.
    """
    return extract_geotiff_info(file_path=image_path, *args, **kwargs)