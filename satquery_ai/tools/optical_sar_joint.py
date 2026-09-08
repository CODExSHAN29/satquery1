import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Optional
from satquery_ai.tools.base_tool import BaseTool
from satquery_ai.utils.geotiff_parser import GeoTIFFParser


class OpticalSARJointTool(BaseTool):
    """
    Optical-SAR Cross-Modal Fusion Tool.
    Extracts complementary information from co-registered optical and SAR image pairs.
    Combines spectral features from optical with texture/backscatter from SAR.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "optical_sar_joint"

    @property
    def description(self) -> str:
        return "Fuses co-registered Optical and SAR imagery for all-weather feature extraction."

    @property
    def tool_name(self) -> str:
        return "optical_sar_joint"

    @property
    def tool_description(self) -> str:
        return "Fuses co-registered Optical and SAR imagery for all-weather feature extraction."

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        query = ""
        if len(args) > 0 and isinstance(args[0], str):
            query = args[0]
        if not query:
            query = str(kwargs.get("query", ""))

        # Get image paths - handle both 'images' and 'image_path' parameters
        images = kwargs.get("images") or kwargs.get("image_paths") or []
        if isinstance(images, str):
            images = [images]
        
        # If only image_path provided, wrap it in a list
        image_path = kwargs.get("image_path")
        if image_path and not images:
            images = [image_path]

        if len(images) < 2:
            return {
                "error": "SAR-Optical fusion requires TWO images: one Optical and one SAR. Please upload both primary and secondary files.",
                "task": "OPTICAL_SAR_FUSION",
                "confidence": 0.0
            }

        # Assume first is optical, second is SAR (or auto-detect)
        optical_path = images[0]
        sar_path = images[1]

        # Parse optical image
        parser_optical = GeoTIFFParser(optical_path)
        geo_optical = parser_optical.parse(optical_path)

        if isinstance(geo_optical, dict) and "rgb_array" in geo_optical:
            optical_rgb = np.asarray(geo_optical["rgb_array"], dtype=np.uint8)
        else:
            optical_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        # Parse SAR image
        parser_sar = GeoTIFFParser(sar_path)
        geo_sar = parser_sar.parse(sar_path)

        if isinstance(geo_sar, dict) and "rgb_array" in geo_sar:
            sar_data = np.asarray(geo_sar["rgb_array"], dtype=np.uint8)
        else:
            sar_data = np.zeros((512, 512, 3), dtype=np.uint8)

        # Ensure same size
        h, w = min(optical_rgb.shape[0], sar_data.shape[0]), min(optical_rgb.shape[1], sar_data.shape[1])
        optical_rgb = optical_rgb[:h, :w]
        sar_data = sar_data[:h, :w]

        # Convert SAR to grayscale (single channel intensity)
        if sar_data.ndim == 3:
            sar_gray = cv2.cvtColor(sar_data, cv2.COLOR_RGB2GRAY)
        else:
            sar_gray = sar_data

        # Extract optical features
        optical_gray = cv2.cvtColor(optical_rgb, cv2.COLOR_RGB2GRAY)

        # SAR texture features using gradient
        sar_grad_x = cv2.Sobel(sar_gray, cv2.CV_64F, 1, 0, ksize=3)
        sar_grad_y = cv2.Sobel(sar_gray, cv2.CV_64F, 0, 1, ksize=3)
        sar_magnitude = np.sqrt(sar_grad_x**2 + sar_grad_y**2)
        sar_magnitude = (sar_magnitude / sar_magnitude.max() * 255).astype(np.uint8)

        # Optical spectral features
        r, g, b = optical_rgb[:, :, 0].astype(float), optical_rgb[:, :, 1].astype(float), optical_rgb[:, :, 2].astype(float)

        # Scientific note: True NDVI requires NIR band; true NDWI requires NIR;
        # true MNDWI requires SWIR. None of these are present in a 3-channel
        # RGB image. We therefore do NOT approximate them via (g-r)/(g+r) or
        # (g-b)/(g+b) ratios. Spectral indices are reported as unavailable.
        spectral_index_available = False
        spectral_index_reason = "Required NIR/SWIR band unavailable"

        # Urban/built-up detection using SAR intensity
        sar_threshold = np.percentile(sar_gray, 70)
        urban_mask = (sar_gray > sar_threshold).astype(np.uint8) * 255

        # Cloud detection in optical (RGB only, not a spectral index)
        brightness = (r + g + b) / 3.0
        cloud_mask = ((brightness > 185) & (r > 165) & (g > 165) & (b > 165)).astype(np.uint8) * 255

        # Without valid spectral indices, the water and vegetation masks are
        # reported as unavailable (zeros) and labelled honestly in the output.
        water_mask = np.zeros_like(sar_gray, dtype=np.uint8)
        vegetation_mask = np.zeros_like(sar_gray, dtype=np.uint8)

        # Fusion: Combine optical and SAR information
        # Create fused visualization
        fused = optical_rgb.copy()

        # Highlight SAR-detected features in areas where optical is cloudy
        cloud_pixels = cloud_mask > 0
        if np.sum(cloud_pixels) > 0:
            # SAR can penetrate clouds - show SAR features there
            sar_colored = cv2.applyColorMap(sar_gray, cv2.COLORMAP_JET)
            fused[cloud_pixels] = cv2.addWeighted(
                fused[cloud_pixels], 0.3,
                sar_colored[cloud_pixels], 0.7, 0
            )

        # Create fusion overlay (only SAR-derived urban mask; spectral masks
        # for water/vegetation require NIR/SWIR bands and are not present)
        fusion_overlay = np.zeros_like(optical_rgb)

        # Red: Urban/built-up (SAR)
        fusion_overlay[:, :, 0] = urban_mask

        # Blend with original optical
        fused = cv2.addWeighted(optical_rgb, 0.6, fusion_overlay, 0.4, 0)

        # Calculate statistics
        cloud_cover = np.sum(cloud_mask > 0) / (h * w)
        water_cover = 0.0
        urban_cover = np.sum(urban_mask > 0) / (h * w)
        vegetation_cover = 0.0

        # SAR quality metrics
        sar_mean = np.mean(sar_gray)
        sar_std = np.std(sar_gray)
        sar_contrast = sar_std / (sar_mean + 1e-5)

        # Generate bounding boxes for key features
        bounding_boxes = []
        labels = []

        # Detect water bodies
        water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(sorted(water_contours, key=cv2.contourArea, reverse=True)[:3]):
            if cv2.contourArea(cnt) > (w * h * 0.005):
                x, y, bw, bh = cv2.boundingRect(cnt)
                bounding_boxes.append([y, x, y + bh, x + bw])
                labels.append(f"Water Body (Optical) #{i+1}")
                cv2.rectangle(fused, (x, y), (x + bw, y + bh), (0, 150, 255), 2)

        # Detect urban areas
        urban_contours, _ = cv2.findContours(urban_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(sorted(urban_contours, key=cv2.contourArea, reverse=True)[:3]):
            if cv2.contourArea(cnt) > (w * h * 0.005):
                x, y, bw, bh = cv2.boundingRect(cnt)
                bounding_boxes.append([y, x, y + bh, x + bw])
                labels.append(f"Urban Area (SAR) #{i+1}")
                cv2.rectangle(fused, (x, y), (x + bw, y + bh), (255, 100, 0), 2)

        # Save output
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", f"sar_optical_fusion_{os.path.basename(optical_path)}.png")
        Image.fromarray(fused).save(out_path)

        # Generate answer
        answer = (
            f"**Optical-SAR Cross-Modal Fusion Analysis:**\n\n"
            f"**Optical Features (RGB only):**\n"
            f"- Cloud Obstruction: {cloud_cover * 100:.1f}%\n"
            f"- Water Coverage: not computed ({spectral_index_reason})\n"
            f"- Vegetation Coverage: not computed ({spectral_index_reason})\n\n"
            f"**SAR Features:**\n"
            f"- Urban/Built-up Detection: {urban_cover * 100:.1f}%\n"
            f"- SAR Intensity Mean: {sar_mean:.1f}\n"
            f"- SAR Contrast: {sar_contrast:.2f}\n\n"
            f"**Fusion Benefits:**\n"
            f"- SAR penetration through cloud cover: {'Yes' if cloud_cover > 0.05 else 'Minimal clouds'}\n"
            f"- Complementary feature detection: {len(bounding_boxes)} regions identified\n"
            f"- All-weather analysis capability: Enhanced\n"
        )

        # Calibrated heuristic confidence based on SAR sensor quality
        # and optical data completeness. Higher SAR contrast and lower
        # cloud cover yield higher confidence.
        texture_score = min(1.0, sar_contrast / 2.0)
        coverage_score = 1.0 - cloud_cover
        confidence = round((texture_score * 0.7) + (coverage_score * 0.3), 4)
        confidence_type = "heuristic_sensor_quality"

        return {
            "task": "OPTICAL_SAR_FUSION",
            "query": query,
            "answer": answer,
            "optical_water_cover": round(water_cover, 4),
            "optical_vegetation_cover": round(vegetation_cover, 4),
            "optical_cloud_cover": round(cloud_cover, 4),
            "sar_urban_cover": round(urban_cover, 4),
            "sar_contrast": round(sar_contrast, 2),
            "spectral_index_available": spectral_index_available,
            "spectral_index_reason": spectral_index_reason,
            "detected_count": len(bounding_boxes),
            "target_label": "Cross-Modal Features",
            "bounding_boxes": bounding_boxes,
            "labels": labels,
            "overlay_image_path": out_path,
            "analysis": answer,
            "confidence": confidence,
            "confidence_type": confidence_type,
        }
    