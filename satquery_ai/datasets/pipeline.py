import os
import json
from typing import List, Dict, Any


class UnifiedDatasetPipeline:
    """
    Unified dataset manager for BigEarthNet, VRSBench, RSVQA, and CDVQA.
    Converts raw benchmark datasets into standard VLM instruction-tuning format.
    """

    def __init__(self, data_root: str):
        self.data_root = data_root

    def load_bigearthnet_manifest(self) -> List[Dict[str, Any]]:
        manifest_path = os.path.join(self.data_root, "bigearthnet", "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return []

    def load_cdvqa_manifest(self) -> List[Dict[str, Any]]:
        manifest_path = os.path.join(self.data_root, "cdvqa", "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return []

    def load_vrsbench_manifest(self) -> List[Dict[str, Any]]:
        manifest_path = os.path.join(self.data_root, "vrsbench", "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return []