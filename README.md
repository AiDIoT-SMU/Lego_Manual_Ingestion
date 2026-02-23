# LEGO Assembly - Refactored

A clean, modular system for processing LEGO instruction manuals using Vision-Language Models (VLM).

## Overview

This refactored version focuses on three core tasks:
1. **VLM Ingestion**: Extract assembly steps with bounding boxes from instruction PDFs
2. **Image Cropping**: Automatically crop parts and subassemblies using VLM-detected bounding boxes
3. **Frontend Display**: View steps and parts catalog with cropped images

## Features

- ✅ PDF to structured JSON extraction with Gemini Robotic ER 1.5 Preview VLM
- ✅ Automatic bounding box detection for parts and subassemblies
- ✅ Image cropping and organized storage
- ✅ RESTful API with FastAPI
- ✅ Clean, modular architecture
- ✅ Single VLM model (no fallbacks) for simplicity
- ✅ JSON file-based storage (no vector database)
- ✅ URL-based PDF download (direct links, LEGO CDN URLs)
- ✅ Image folder input with automatic page renaming
- ✅ Optional image preprocessing (contrast/sharpness enhancement, multi-step page segmentation)
- ❌ No RAG, embeddings, graph structures, or video analysis

## Architecture

```
Input (PDF / Image Folder / URL) → VLM Extraction → Image Cropping → JSON Output → Frontend Display
```

## Installation

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Gemini API key

### Setup

1. Install dependencies:
```bash
uv sync
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Usage

### 1. Run the Backend API

```bash
uv run python -m backend.main
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 1b. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

Navigate to `http://localhost:3000/ingest` to use the ingestion UI.

### 2. Process a Manual

**Option A: Upload a PDF via API**

```bash
curl -X POST "http://localhost:8000/api/ingest/pdf" \
  -F "manual_id=6262059" \
  -F "instruction_pages=[13,14,15,16,17,18,19,20]" \
  -F "pdf_file=@/path/to/manual.pdf"
```

**Option B: Provide a URL**

```bash
curl -X POST "http://localhost:8000/api/ingest/url" \
  -F "manual_id=6262059" \
  -F "url=https://www.lego.com/cdn/product-assets/.../6262059.pdf" \
  -F "instruction_pages=[13,14,15,16,17,18,19,20]"
```

**Option C: Upload images via API**

```bash
curl -X POST "http://localhost:8000/api/ingest/images" \
  -F "manual_id=6262059" \
  -F "images=@page1.png" \
  -F "images=@page2.png"
```

**Option D: Via Python script**

```python
from pathlib import Path
from config.settings import get_settings
from ingestion.pipeline import IngestionPipeline

settings = get_settings()
pipeline = IngestionPipeline(settings)

# From PDF
result = pipeline.process_manual(
    manual_id="6262059",
    pdf_path=Path("/path/to/manual.pdf"),
    instruction_pages=[13, 14, 15, 16, 17, 18, 19, 20]
)

# From URL
result = pipeline.process_url(
    manual_id="6262059",
    url="https://www.lego.com/cdn/product-assets/.../6262059.pdf",
    instruction_pages=[13, 14, 15, 16, 17, 18, 19, 20]
)

# From image folder — place images in data/manuals/{manual_id}/source/
result = pipeline.process_image_directory(
    manual_id="6262059",
    image_dir=Path("data/manuals/6262059/source/")
)

print(f"Extracted {len(result.steps)} steps")
```

### 3. Query the API

**List manuals:**
```bash
curl http://localhost:8000/api/manuals
```

**Get all steps:**
```bash
curl http://localhost:8000/api/manuals/6262059/steps
```

**Get parts catalog:**
```bash
curl http://localhost:8000/api/manuals/6262059/parts
```

**Access cropped images:**
```
http://localhost:8000/images/6262059/parts/step_1_part_0.png
http://localhost:8000/images/6262059/subassemblies/step_1_subassembly_0.png
```

## Directory Structure

```
lego_assembly_refactored/
├── config/              # Configuration management
├── ingestion/           # Core processing pipeline
│   ├── pipeline.py          # Orchestrates all input types
│   ├── pdf_processor.py     # PDF/image page extraction
│   ├── url_handler.py       # PDF download from URLs
│   ├── manual_input_handler.py  # Image preprocessing utilities
│   ├── vlm_extractor.py     # Gemini VLM step extraction
│   ├── image_cropper.py     # Bounding box image cropping
│   └── schemas.py           # Pydantic data models
├── backend/             # FastAPI application
│   ├── routes/          # API endpoints
│   └── services/        # Business logic
├── frontend/            # Next.js frontend
│   ├── app/
│   │   ├── ingest/      # Ingestion UI (URL or image upload)
│   │   ├── steps/       # Step viewer
│   │   └── parts/       # Parts catalog viewer
│   └── lib/
│       └── api.ts       # Typed API client
├── data/                # Generated data
│   ├── manuals/         # Uploaded PDFs and source images
│   ├── processed/       # JSON outputs
│   └── cropped/         # Cropped part/subassembly images
├── prompts/             # VLM prompt templates
├── pyproject.toml       # Project dependencies (managed by uv)
└── uv.lock              # Locked dependency versions
```

## Data Flow

1. **Input** → PDF upload, URL download (temp file), or image folder placed in `data/manuals/{manual_id}/`
2. **Page Extraction** → Pages saved as `page_001.png`, `page_002.png`, etc. in `data/manuals/{manual_id}/`
3. **VLM Extraction** → Gemini analyzes each page, outputs steps with bounding boxes
4. **JSON Output** → `data/processed/{manual_id}/extraction.json`
5. **Image Cropping** → Parts/subassemblies cropped using bounding boxes
6. **Enhanced JSON** → `data/processed/{manual_id}/enhanced.json` with image paths
7. **Frontend Access** → API serves data and images

> **Image folder input:** Place your images in any directory and call `process_image_directory()` or `POST /api/ingest/images`. Images are automatically copied to `data/manuals/{manual_id}/` and renamed `page_001.png`, `page_002.png`, etc.

## Output Format

### extraction.json
```json
{
  "manual_id": "6262059",
  "steps": [
    {
      "step_number": 1,
      "parts_required": [
        {
          "description": "red 2x4 brick",
          "bounding_box": {"x": 100, "y": 150, "width": 80, "height": 60},
          "cropped_image_path": null
        }
      ],
      "subassemblies": [...],
      "actions": ["Attach red brick to base"],
      "source_page_path": "data/manuals/6262059/page_013.png",
      "notes": null
    }
  ]
}
```

### enhanced.json
Same structure as extraction.json, but with `cropped_image_path` fields populated.

## Configuration

Edit `.env` to customize:

- `GEMINI_API_KEY`: Your Gemini API key
- `VLM_MODEL`: VLM model to use (default: gemini/gemini-robotics-er-1.5-preview)
- `API_PORT`: Backend server port (default: 8000)
- `DATA_DIR`: Base directory for all data (default: ./data)

## Development

### Running Tests
```bash
uv run pytest tests/
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

## Differences from Original

| Feature | Original | Refactored |
|---------|----------|------------|
| Part descriptions | Multi-attribute object | Single-line string |
| Bounding boxes | Not included | Required |
| VLM models | Primary + 2 fallbacks | Single model only |
| Context system | BuildMemory, TokenBudget | None (stateless) |
| Backend features | RAG, embeddings, graph, video | JSON serving only |
| Frontend | Chat + video player | Steps + parts catalog |
| Storage | ChromaDB vector store | JSON files |

## Troubleshooting

**Issue: "PyMuPDF not installed"**
```bash
uv add pymupdf
```

**Issue: "GEMINI_API_KEY not set"**
- Ensure `.env` file exists with valid API key
- Check that the key has correct permissions

**Issue: VLM extraction fails**
- Check API key is valid
- Ensure you have sufficient API quota
- Try reducing the number of pages processed at once

## License

MIT

## Support

For issues and questions, please create an issue on GitHub.
