import os
import json
from typing import List, Dict, Any


class BigEarthNetPreparer:
    """
    Formats BigEarthNet Sentinel-2 (Optical) and Sentinel-1 (SAR) imagery
    and land-cover class annotations into VLM instruction-tuning format.
    """

    @staticmethod
    def format_sample(image_path: str, land_cover_labels: List[str]) -> Dict[str, Any]:
        """Converts a BigEarthNet sample into an instruction-tuning JSON item."""
        labels_str = ", ".join(land_cover_labels)
        return {
            "image": image_path,
            "conversations": [
                {
                    "role": "user",
                    "value": "<image>\nDescribe the land cover categories present in this multisensor remote sensing patch."
                },
                {
                    "role": "assistant",
                    "value": f"This satellite patch contains the following land cover types: {labels_str}."
                }
            ]
        }

    @staticmethod
    def create_dataset_json(data_dir: str, output_json: str):
        """Scans dataset directory and builds a unified training manifest."""
        samples = []
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith(('.tif', '.png', '.jpg')):
                        img_path = os.path.join(root, f)
                        # Example sample formatting
                        sample = BigEarthNetPreparer.format_sample(img_path, ["Arable land", "Coniferous forest"])
                        samples.append(sample)

        with open(output_json, 'w') as f_out:
            json.dump(samples, f_out, indent=2)
        print(f"[BigEarthNet] Saved {len(samples)} formatted samples to {output_json}")