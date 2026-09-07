"""Standalone GeoChat test — Phase 1B/1C/1D."""
import time, sys, os

def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from satquery_ai.models.geochat_service import GeoChatService

    IMG = "uploads/ROIs1868_summer_s1_59_p10.png"

    svc = GeoChatService()
    print("=== GeoChat environment ===")
    print("model_id:", svc.model_id)
    print("device:", svc.device)
    print("torch available:", __import__("torch").__version__)

    # Load
    ok = svc.load()
    print("load ok:", ok)
    if not ok:
        sys.exit(1)

    print("\n=== PROMPT 1: Describe satellite image ===")
    t0 = time.time()
    raw1 = svc.vqa(IMG, "Describe this satellite image.", max_tokens=128)
    t1 = time.time()
    print("time_ms:", round((t1-t0)*1000,1))
    print("RAW OUTPUT:")
    print(repr(raw1))

    print("\n=== PROMPT 2: What land cover is visible? ===")
    t0 = time.time()
    raw2 = svc.vqa(IMG, "What land cover is visible?", max_tokens=128)
    t1 = time.time()
    print("time_ms:", round((t1-t0)*1000,1))
    print("RAW OUTPUT:")
    print(repr(raw2))

    print("\n=== PROMPT 3: Locate the river ===")
    t0 = time.time()
    raw3 = svc.ground(IMG, "Locate the river.", max_tokens=64)
    t1 = time.time()
    print("time_ms:", round((t1-t0)*1000,1))
    print("RAW OUTPUT (grounding):")
    print(repr(raw3))

    # Grounding parser inspection
    print("\n=== GROUNDING PARSER INSPECTION ===")
    if raw3:
        import re
        boxes = []
        for m in re.finditer(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", raw3):
            boxes.append([int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))])
        print("boxes found in raw:", boxes)
        # Try to detect coordinate syntax / normalization
        if boxes:
            print("coordinate syntax detected (normalized or pixel?)")
        else:
            print("no box syntax in output; grounding_available = false")

    print("\n=== STATUS ===")
    print("caption:", "PASS" if (raw1 and not raw1.startswith("[")) else "FAIL")
    print("vqa:", "PASS" if (raw2 and not raw2.startswith("[")) else "FAIL")
    print("grounding:", "PARTIAL (text returned, no coords)" if raw3 else "FAIL")


if __name__ == "__main__":
    main()
