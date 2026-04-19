# LEGO Assembly System

A comprehensive system for LEGO assembly processing, featuring VLM-based instruction extraction and digital twin-based error detection for assembly validation.

## Overview

This system combines two powerful capabilities:

1. **VLM Ingestion Pipeline**: Extract assembly steps from instruction manuals using Vision-Language Models
2. **Digital Twin System**: Build 3D digital representations of LEGO assemblies for error detection and visualization

## Features

### VLM Ingestion
- ✅ PDF to structured JSON extraction with Gemini Robotics ER 1.5 Preview VLM
- ✅ Automatic bounding box detection for parts and subassemblies
- ✅ Image cropping and organized storage
- ✅ RESTful API with FastAPI
- ✅ URL-based PDF download (direct links, LEGO CDN URLs)
- ✅ Image folder input with automatic page renaming
- ✅ Optional image preprocessing (contrast/sharpness enhancement)

### Video Assembly Verification
- ✅ Video upload and frame extraction (OpenCV, every 50 frames)
- ✅ VLM-based step detection with multi-image comparison
- ✅ Part detection with confidence scores and temporal smoothing
- ✅ Step timeline with clickable timestamps
- ✅ Parts checklist with first-seen frame references
- ✅ Background processing with polling status endpoint

### Digital Twin & CAD Processing
- ✅ LDraw file parsing and processing
- ✅ High-fidelity 3D mesh generation (27,540+ vertices per brick)
- ✅ Per-brick identification and tracking
- ✅ Brick library system for efficient geometry reuse
- ✅ 3D visualization with Open3D
- ✅ Structured digital twin database with pose information
- ✅ Multi-step assembly support

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEGO Assembly System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  VLM Pipeline                    Digital Twin System             │
│  ─────────────                   ────────────────                │
│  PDF/Images                      LDraw Files                     │
│      ↓                                ↓                          │
│  VLM Extraction                  CAD Processing                  │
│      ↓                                ↓                          │
│  Bounding Boxes                  Brick Library                   │
│      ↓                                ↓                          │
│  Image Cropping                  Digital Twin DB                 │
│      ↓                                ↓                          │
│  JSON Output                     3D Visualization                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Gemini API key (for VLM features)
- LDraw library (for CAD processing)

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

3. (Optional) Download LDraw library for CAD processing:
```bash
# Download from https://www.ldraw.org/library/
# Extract to data/ldraw_library/ldraw/
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

Four input modes are available in the UI:
- **URL** — paste a direct link to a PDF; optionally specify instruction pages (e.g. `13-20, 25`)
- **Upload PDF** — drag & drop or select a local PDF file; optionally specify instruction pages
- **Upload Images** — upload individual page images and tag each as Instruction, Final Assembly, or Parts Catalog
- **Video** — upload an assembly video to verify step progress and detect part placement (manual must be ingested first)

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

### Digital Twin System

#### 1. Build Brick Library (One-Time Setup)

```bash
uv run python scripts/build_brick_library.py
```

This generates:
- High-fidelity 3D meshes (.obj) for each unique brick type
- Metadata file with brick information
- Reusable library stored in `data/brick_library/`

#### 2. Build Digital Twin

```bash
uv run python scripts/build_digital_twin.py
```

This creates:
- Per-step JSON files with brick poses and references
- Master digital twin database
- Output in `data/processed/123456/digital_twin/`

#### 3. Visualize Assembly

```bash
# View specific step
uv run python scripts/view_digital_twin.py 1

# View all steps sequentially
uv run python scripts/view_digital_twin.py all

# Default (step 1)
uv run python scripts/view_digital_twin.py
```

**Visualization Controls:**
- Mouse drag: Rotate view
- Scroll: Zoom in/out
- Ctrl + Mouse: Pan
- Q or ESC: Close window

## Directory Structure

```
lego_assembler/
├── backend/                 # FastAPI application
│   ├── routes/             # API endpoints
│   │   ├── ingestion.py   # Ingest routes
│   │   ├── video.py       # Video upload & analysis routes
│   │   └── ...
│   └── services/           # Business logic
│       ├── video_processor.py  # Frame extraction (OpenCV)
│       ├── video_analyzer.py   # VLM orchestration
│       └── ...
├── frontend/               # Next.js frontend
│   ├── app/
│   │   ├── ingest/        # Ingestion UI (URL/PDF/images/video)
│   │   ├── steps/         # Step viewer
│   │   ├── parts/         # Parts catalog
│   │   └── video-verify/  # Video upload & results viewer
│   └── lib/
│       └── api.ts         # Typed API client
├── prompts/               # VLM prompt templates
│   ├── video_step_detection.txt
│   └── video_part_detection.txt
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

### Digital Twin Pipeline
1. **Input** → LDraw (.ldr) files
2. **Library Building** → Generate reusable brick meshes
3. **Digital Twin Creation** → Per-step metadata with poses
4. **Visualization** → 3D rendering with Open3D

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

### Digital Twin JSON
```json
{
  "step_number": 1,
  "num_bricks": 9,
  "bricks": [
    {
      "brick_id": 0,
      "part_number": "4204.dat",
      "color_id": 15,
      "position": [-40.0, -24.0, -40.0],
      "rotation_matrix": [[1,0,0], [0,1,0], [0,0,1]],
      "pose_4x4": [[1,0,0,-40], [0,1,0,-24], [0,0,1,-40], [0,0,0,1]],
      "rotation_angles_deg": {"roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0},
      "geometry_reference": {
        "mesh_file": "4204_c15.obj",
        "library_key": "4204.dat_15"
      }
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

### Video Analysis
- `POST /api/video/upload` - Upload video and trigger background analysis
- `GET /api/video/analysis/{manual_id}/{video_id}` - Poll for results or retrieve full analysis
- `GET /api/video/list/{manual_id}` - List all analyzed videos for a manual

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

### High-Fidelity Mesh Generation
- Full geometric detail including studs, tubes, and embossing
- 27,540+ vertices per complex brick (vs. 34 vertices with merged approach)
- Recursive sub-part loading for accurate representations

### Per-Brick Tracking
- Individual identification for each brick instance
- 6DoF pose estimation (position + rotation)
- Links to reusable brick library for efficiency

### Brick Library System
- Generate once, reuse everywhere
- Stores unique part+color combinations
- Metadata includes vertices, faces, and file references

## Future Enhancements

- [ ] Real-time error detection with camera integration
- [ ] ICP-based pose refinement
- [ ] AR guidance overlays
- [ ] Multi-user collaborative assembly
- [ ] Temporal tracking across frames
- [ ] YOLO-based brick detection

## Performance

**Digital Twin Generation:**
- Brick library: ~5 minutes (one-time setup)
- Digital twin metadata: <1 second per assembly

**Runtime:**
- API response time: <100ms
- VLM extraction: ~2-5 seconds per page
- 3D visualization: Real-time (60+ FPS)

## Troubleshooting

**Issue: "PyMuPDF not installed"**
```bash
uv add pymupdf
```

**Issue: "GEMINI_API_KEY not set"**
- Ensure `.env` file exists with valid API key
- Check key permissions and quota

**Issue: "LDraw library not found"**
- Download from https://www.ldraw.org/library/
- Extract to `data/ldraw_library/ldraw/`

**Issue: VLM extraction fails**
- Verify API key is valid
- Check API quota
- Reduce number of pages processed at once

**Issue: Mesh generation fails**
- Ensure LDraw library is properly installed
- Check that .dat files exist in ldraw/parts/

## Technologies Used

- **Backend**: FastAPI, Python 3.12
- **Frontend**: Next.js, React, TypeScript
- **VLM**: Google Gemini (Robotics ER 1.5 Preview)
- **3D Processing**: Trimesh, Open3D, NumPy
- **CAD**: LDraw format parsing
- **Package Management**: uv

## Research Applications

This system is designed for research in:
- Robotic assembly assistance
- AR-guided manufacturing
- Error detection in manual assembly
- Digital twin applications
- Computer vision for object tracking

## License

MIT

## Support

For issues and questions, please create an issue on GitHub.
