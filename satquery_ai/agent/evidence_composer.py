"""Evidence composition: merge specialist outputs into unified FusedResponse."""
from typing import List, Dict, Any
from satquery_ai.schemas.result_schema import FusedResponse, EvidenceItem


def compose_evidence(outputs: List[Dict[str, Any]]) -> FusedResponse:
    """Merge specialist outputs into unified result."""
    evidence = []
    all_classes = []
    avg_conf = 0.0
    specialist = "fusion"
    for o in outputs:
        spec = o.get("model", o.get("specialist", "unknown"))
        if isinstance(spec, str):
            specialist = spec
        conf = o.get("confidence", 0.0)
        if isinstance(conf, (int, float)):
            avg_conf += float(conf)
        if "predicted_classes" in o:
            all_classes.extend(o["predicted_classes"])
        for item in o.get("spatial_evidence", o.get("bounding_boxes", [])):
            evidence.append({
                "specialist": specialist,
                "label": o.get("analysis", "").split()[0] if o.get("analysis") else "feature",
                "box": item if isinstance(item, list) else [],
                "confidence": float(o.get("confidence", 0.0)) if isinstance(o.get("confidence"), (int, float)) else 0.5,
            })
    avg_conf = avg_conf / max(len(outputs), 1)
    # Deduplicate predicted classes
    unique_classes = list(dict.fromkeys(str(c) for c in all_classes))
    return FusedResponse(
        answer=" ".join([o.get("answer", o.get("analysis", "")) for o in outputs]),
        spatial_evidence=evidence,
        confidence=round(min(0.95, avg_conf), 2),
        specialist_used=specialist,
        sub_task="sar_optical_fusion",
        predicted_classes=unique_classes,
        bounding_boxes=[ev.get("box", []) for ev in evidence],
    )
