# 🛰️ SatQuery AI — ISRO Agentic Vision-Language Satellite Assistant

An ISRO-compliant, agentic vision-language AI system for remote sensing and GeoTIFF satellite imagery.

## Features

- **Task Modes**: Multi-Object Text Grounding, Single-Scene VQA, Bi-Temporal Change Detection, Optical + SAR Joint Analysis
- **Agentic Routing**: Automatic specialist model selection based on query content and image count
- **Vision-Language Backend**: Qwen2-VL with LoRA fine-tuning on BigEarthNet, RSVQA, CDVQA
- **GeoTIFF Parsing**: Multi-band Sentinel-2 / Landsat-9 / ISRO Cartosat-3 support via `rasterio`
- **Execution Trace**: ISRO-compliant auditable JSON trace with confidence scores
- **PDF Reporting**: Downloadable execution report for ISRO submission
## DEMO

Live demo :- https://satmax.streamlit.app/


## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended) or CPU-only mode
- `git`, `pip`

### Install
```bash
pip install -r requirements.txt
```

### Run Streamlit App (default)
```bash
python -m streamlit run app.py --server.headless true --server.port 8501
```

### Run FastAPI Backend
```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## Project Structure
```
newsat/
├── app.py                    # Streamlit frontend
├── api_server.py             # FastAPI backend
├── engine.py                 # Master SatQueryEngine
├── download_datasets.py      # Dataset downloader
├── train_vlm.py              # Fine-tuning script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build
├── docker-compose.yml        # Multi-service orchestration
├── groundingdino/            # Grounding DINO config + weights
├── satquery_ai/              # Core package
│   ├── agent/                # AgentController, TraceLogger, QueryClassifier
│   ├── models/               # BaseVLMLoader (Qwen2-VL / BLIP-2)
│   ├── tools/                # Specialist tool implementations
│   ├── utils/                # GeoTIFF parser, ReportGenerator, Visualizer
│   ├── datasets/             # Dataset prep pipelines
│   └── __init__.py
├── datasets/raw/             # BigEarthNet, CDVQA, VRSBench manifests
├── uploads/                  # User-uploaded satellite files
├── outputs/                  # Generated overlays
├── logs/traces/              # Execution trace JSONs
└── tests/                    # Benchmark test suite
```

## ISRO Compliance Status

| Deliverable | Status |
|---|---|
| Interactive GUI | ✅ Complete |
| Agentic Backend | ✅ Complete |
| Text-Guided Grounding | ✅ Complete |
| Bi-Temporal Change | ✅ Complete |
| SAR-Optical Fusion | ✅ Complete |
| Execution Trace | ✅ Complete |
| VLM Fine-Tuning | ⏳ In progress |
| Public Benchmarks | ⏳ In progress |
| ISRO Problem Statement | 📄 See `ISRO_COMPLIANCE_AUDIT.md` |

## License
ISRO Problem Statement Compliance — See `ISRO_COMPLIANCE_AUDIT.md`

## Contact
SatQuery AI — Vision-Language Satellite Assistant
