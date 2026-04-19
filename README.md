# LEGO Assembly System

A comprehensive system for LEGO assembly processing, featuring VLM-based instruction extraction from LEGO instruction manuals.

## Overview

This system provides a powerful VLM Ingestion Pipeline that extracts assembly steps from instruction manuals using Vision-Language Models.

## Features

### VLM Ingestion
- ✅ PDF to structured JSON extraction with Gemini Robotics ER 1.5 Preview VLM
- ✅ Automatic bounding box detection for parts and subassemblies
- ✅ Image cropping and organized storage
- ✅ RESTful API with FastAPI
- ✅ URL-based PDF download (direct links, LEGO CDN URLs)
- ✅ Image folder input with automatic page renaming
- ✅ Optional image preprocessing (contrast/sharpness enhancement)

## Architecture

```
┌─────────────────────────────────────────────┐
│         LEGO Assembly System                │
├─────────────────────────────────────────────┤
│                                             │
│  VLM Pipeline                               │
│  ─────────────                              │
│  PDF/Images                                 │
│      ↓                                      │
│  VLM Extraction                             │
│      ↓                                      │
│  Bounding Boxes                             │
│      ↓                                      │
│  Image Cropping                             │
│      ↓                                      │
│  JSON Output                                │
│                                             │
└─────────────────────────────────────────────┘
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

### VLM Ingestion Pipeline

#### 1. Start the Backend API

```bash
uv run python -m backend.main
```

API available at: `http://localhost:8000`
Documentation: `http://localhost:8000/docs`

#### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:3000`

#### 3. Process Instruction Manuals

**Via Web UI:**
Navigate to `http://localhost:3000/ingest`

Three input modes are available in the UI:
- **URL** — paste a direct link to a PDF; optionally specify instruction pages (e.g. `13-20, 25`)
- **Upload PDF** — drag & drop or select a local PDF file; optionally specify instruction pages
- **Upload Images** — upload individual page images and tag each as Instruction, Final Assembly, or Parts Catalog

**Via API:**

```bash
# Upload PDF
curl -X POST "http://localhost:8000/api/ingest/pdf" \
  -F "manual_id=6262059" \
  -F "instruction_pages=[13,14,15,16,17,18,19,20]" \
  -F "pdf_file=@/path/to/manual.pdf"

# From URL
curl -X POST "http://localhost:8000/api/ingest/url" \
  -F "manual_id=6262059" \
  -F "url=https://www.lego.com/cdn/product-assets/.../6262059.pdf" \
  -F "instruction_pages=[13,14,15,16,17,18,19,20]"

# Upload images
curl -X POST "http://localhost:8000/api/ingest/images" \
  -F "manual_id=6262059" \
  -F "images=@page1.png" \
  -F "images=@page2.png"
```

## Directory Structure

```
lego_assembler/
├── backend/                 # FastAPI application
│   ├── routes/             # API endpoints
│   │   ├── ingestion.py   # Ingest routes
│   │   ├── steps.py       # Step routes
│   │   └── parts.py       # Parts catalog routes
│   └── services/           # Business logic
│       └── data_service.py  # Data management
├── frontend/               # Next.js frontend
│   ├── app/
│   │   ├── ingest/        # Ingestion UI (URL/PDF/images)
│   │   ├── steps/         # Step viewer
│   │   └── parts/         # Parts catalog
│   └── lib/
│       └── api.ts         # Typed API client
├── prompts/               # VLM prompt templates
├── ingestion/              # VLM processing pipeline
│   ├── pipeline.py        # Main orchestrator
│   ├── vlm_extractor.py   # Gemini VLM integration
│   ├── image_cropper.py   # Bounding box cropping
│   └── schemas.py         # Data models
├── scripts/                # Utility scripts
│   ├── check_available_gemini_models.py
│   └── clear_ingestion.py
├── config/                 # Configuration
│   └── settings.py
└── data/                   # Generated data
    ├── manuals/           # Uploaded PDFs and images
    ├── processed/         # JSON outputs
    └── cropped/           # Cropped images
```

## Data Flow

### VLM Pipeline
1. **Input** → PDF, URL, or image folder
2. **Page Extraction** → Save as PNG images
3. **VLM Analysis** → Gemini extracts steps with bounding boxes
4. **JSON Output** → `data/processed/{manual_id}/extraction.json`
5. **Image Cropping** → Parts/subassemblies cropped
6. **Enhanced JSON** → `data/processed/{manual_id}/enhanced.json` with paths

## Output Formats

### VLM Extraction JSON
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
          "cropped_image_path": "data/cropped/6262059/parts/step_1_part_0.png"
        }
      ],
      "subassemblies": [...],
      "actions": ["Attach red brick to base"],
      "source_page_path": "data/manuals/6262059/page_013.png"
    }
  ]
}
```

## Configuration

Edit `.env` to customize:

```bash
# Gemini API
GEMINI_API_KEY=your_key_here
VLM_MODEL=gemini/gemini-robotics-er-1.5-preview

# API Settings
API_PORT=8000

# Data Directories
DATA_DIR=./data
```

## API Endpoints

### VLM Ingestion
- `POST /api/ingest/pdf` - Upload PDF
- `POST /api/ingest/url` - Process from URL
- `POST /api/ingest/images` - Upload images
- `GET /api/manuals` - List processed manuals
- `GET /api/manuals/{manual_id}/steps` - Get all steps
- `GET /api/manuals/{manual_id}/parts` - Get parts catalog
- `GET /images/{manual_id}/parts/{filename}` - Serve cropped images

## Development

### Code Structure
- **Modular design**: Separation of concerns across modules
- **Type safety**: Pydantic models for data validation
- **Clean architecture**: API, business logic, and data processing layers

### Project Organization
- `backend/` - FastAPI server and API routes
- `frontend/` - Next.js web application
- `ingestion/` - Core VLM extraction pipeline
- `config/` - Application configuration
- `prompts/` - VLM prompt templates

## Key Features Explained

### VLM-Powered Extraction
- Utilizes state-of-the-art Gemini Robotics ER 1.5 Preview model
- Automatic detection of parts and subassemblies with bounding boxes
- Structured JSON output for easy integration and processing

### Flexible Input Options
- Support for PDF files, direct URLs, and image folders
- Configurable page range selection for targeted processing
- Optional image preprocessing for enhanced extraction quality

## Future Enhancements

- [ ] Enhanced multi-language support for instruction manuals
- [ ] Improved part recognition accuracy
- [ ] Support for additional manual formats
- [ ] Batch processing capabilities
- [ ] Assembly step validation

## Performance

**VLM Extraction:**
- API response time: <100ms
- VLM extraction: ~2-5 seconds per page
- Image processing: Real-time

## Troubleshooting

**Issue: "PyMuPDF not installed"**
```bash
uv add pymupdf
```

**Issue: "GEMINI_API_KEY not set"**
- Ensure `.env` file exists with valid API key
- Check key permissions and quota

**Issue: VLM extraction fails**
- Verify API key is valid
- Check API quota
- Reduce number of pages processed at once

## Technologies Used

- **Backend**: FastAPI, Python 3.12
- **Frontend**: Next.js, React, TypeScript
- **VLM**: Google Gemini (Robotics ER 1.5 Preview)
- **Image Processing**: PIL, PyMuPDF, NumPy
- **Package Management**: uv

## Research Applications

This system is designed for research in:
- Automated instruction extraction
- Vision-language model applications
- Assembly step understanding
- Document processing automation

## License

MIT

## Support

For issues and questions, please create an issue on GitHub.
