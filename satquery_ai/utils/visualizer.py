from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from typing import Dict, Any, List, Tuple


class OverlayVisualizer:
    """
    Renders visual evidence overlays (bounding boxes, change detection masks,
    SAR coherence maps) directly onto satellite PIL images.
    """

    @staticmethod
    def draw_bounding_boxes(image: Image.Image, visual_evidence: Dict[str, Any]) -> Image.Image:
        """Draw clean neon red bounding boxes and text labels onto the satellite image."""
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        w, h = annotated.size

        boxes = visual_evidence.get("boxes", [])
        labels = visual_evidence.get("labels", [])
        scores = visual_evidence.get("scores", [])

        # Dynamic scaling based on image dimensions
        thickness = max(2, int(w / 200))
        font_size = max(12, int(w / 35))

        for i, box in enumerate(boxes):
            # Coordinates normalized 0-1000 scale: [ymin, xmin, ymax, xmax]
            ymin = int((box[0] / 1000.0) * h)
            xmin = int((box[1] / 1000.0) * w)
            ymax = int((box[2] / 1000.0) * h)
            xmax = int((box[3] / 1000.0) * w)

            # Ensure xmax >= xmin and ymax >= ymin
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if ymax < ymin:
                ymin, ymax = ymax, ymin

            label_text = labels[i] if i < len(labels) else "Target Region"
            score = scores[i] if i < len(scores) else 0.90
            caption = f" {label_text} ({int(score * 100)}%) "

            # Draw thick red bounding box
            for offset in range(thickness):
                x0 = max(0, xmin - offset)
                y0 = max(0, ymin - offset)
                x1 = min(w - 1, xmax + offset)
                y1 = min(h - 1, ymax + offset)
                if x1 >= x0 and y1 >= y0:
                    draw.rectangle([x0, y0, x1, y1], outline=(255, 30, 30))

            # Label tag rectangle - ensure y0 <= y1
            tag_height = font_size + 4
            tag_y0 = max(0, ymin - tag_height)
            tag_y1 = ymin
            if tag_y1 <= tag_y0:
                tag_y1 = tag_y0 + tag_height

            text_w = max(40, int(len(caption) * (font_size * 0.6)))
            tag_x0 = xmin
            tag_x1 = min(w - 1, xmin + text_w)
            
            if tag_x1 >= tag_x0 and tag_y1 >= tag_y0:
                draw.rectangle([tag_x0, tag_y0, tag_x1, tag_y1], fill=(255, 30, 30))
                draw.text((tag_x0 + 4, tag_y0 + 2), caption, fill=(255, 255, 255))

        return annotated

    @staticmethod
    def render_change_mask(image_t1: Image.Image, image_t2: Image.Image) -> Image.Image:
        """Renders bi-temporal change detection heatmap overlaid on T2 image."""
        t1 = image_t1.resize((512, 512)).convert("RGB")
        t2 = image_t2.resize((512, 512)).convert("RGB")

        arr1 = np.array(t1, dtype=np.float32)
        arr2 = np.array(t2, dtype=np.float32)

        # Compute Absolute Pixel Difference
        diff = np.abs(arr2 - arr1).mean(axis=2)
        change_mask = diff > 35.0

        # Create RGBA overlay (Red = Significant structural change)
        overlay = np.zeros((512, 512, 4), dtype=np.uint8)
        overlay[change_mask] = [255, 50, 50, 160]

        overlay_img = Image.fromarray(overlay, mode="RGBA")
        blended = Image.alpha_composite(t2.convert("RGBA"), overlay_img)
        return blended.convert("RGB")

    @staticmethod
    def render_optical_sar_fusion(optical_img: Image.Image, sar_img: Image.Image) -> Image.Image:
        """Renders side-by-side co-registered view of Optical + SAR imagery."""
        opt = optical_img.resize((512, 512)).convert("RGB")
        sar = sar_img.resize((512, 512)).convert("RGB")

        canvas = Image.new("RGB", (1024, 512))
        canvas.paste(opt, (0, 0))
        canvas.paste(sar, (512, 0))

        draw = ImageDraw.Draw(canvas)
        draw.line([(512, 0), (512, 512)], fill=(255, 255, 0), width=4)
        draw.rectangle([10, 10, 180, 40], fill=(0, 0, 0))
        draw.text((18, 18), "OPTICAL (Sentinel-2)", fill=(255, 255, 255))

        draw.rectangle([522, 10, 680, 40], fill=(0, 0, 0))
        draw.text((530, 18), "SAR (Sentinel-1)", fill=(255, 255, 255))

        return canvas