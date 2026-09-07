import json
from typing import List, Dict, Any


class RSVQADataPreparer:
    """Formats RSVQA and VRSBench question-answer pairs for VLM fine-tuning."""

    @staticmethod
    def format_rsvqa_entry(image_path: str, question: str, answer: str) -> Dict[str, Any]:
        return {
            "image": image_path,
            "conversations": [
                {
                    "role": "user",
                    "value": f"<image>\n{question}"
                },
                {
                    "role": "assistant",
                    "value": answer
                }
            ]
        }

    @staticmethod
    def convert_raw_rsvqa(raw_json_path: str, output_json_path: str):
        """Converts raw RSVQA JSON annotations into standardized instruction format."""
        formatted_data = []
        try:
            with open(raw_json_path, 'r') as f:
                raw_data = json.load(f)

            for item in raw_data:
                entry = RSVQADataPreparer.format_rsvqa_entry(
                    image_path=item.get("img_path", ""),
                    question=item.get("question", ""),
                    answer=item.get("answer", "")
                )
                formatted_data.append(entry)

            with open(output_json_path, 'w') as f_out:
                json.dump(formatted_data, f_out, indent=2)
            print(f"[RSVQA] Converted {len(formatted_data)} items to {output_json_path}")

        except FileNotFoundError:
            print(f"[RSVQA] File {raw_json_path} not found. Skipped.")