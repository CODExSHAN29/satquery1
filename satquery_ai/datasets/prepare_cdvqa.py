import json
from typing import Dict, Any, List


class CDVQADataPreparer:
    """Formats CDVQA bi-temporal image pair change questions into VLM instruction dataset format."""

    @staticmethod
    def format_cdvqa_entry(t1_path: str, t2_path: str, question: str, answer: str) -> Dict[str, Any]:
        return {
            "images": [t1_path, t2_path],
            "conversations": [
                {
                    "role": "user",
                    "value": f"Image T1: <image>\nImage T2: <image>\nQuestion: {question}"
                },
                {
                    "role": "assistant",
                    "value": answer
                }
            ]
        }

    @staticmethod
    def convert_cdvqa_dataset(raw_cdvqa_json: str, output_json: str):
        formatted = []
        try:
            with open(raw_cdvqa_json, 'r') as f:
                data = json.load(f)

            for item in data:
                entry = CDVQADataPreparer.format_cdvqa_entry(
                    t1_path=item.get("image_t1", ""),
                    t2_path=item.get("image_t2", ""),
                    question=item.get("question", ""),
                    answer=item.get("change_description", "")
                )
                formatted.append(entry)

            with open(output_json, 'w') as f_out:
                json.dump(formatted, f_out, indent=2)
            print(f"[CDVQA] Processed {len(formatted)} change-VQA pairs to {output_json}")

        except FileNotFoundError:
            print(f"[CDVQA] File {raw_cdvqa_json} not found. Skipped.")