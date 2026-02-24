"""
FastAPI Backend for LEGO Assembly Refactored.
Provides REST API for manual ingestion and data serving.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from loguru import logger

from config.settings import get_settings
from backend.routes import ingestion, steps, parts, digital_twin


# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="LEGO Assembly API",
    description="API for LEGO instruction manual processing and assembly guidance",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving cropped images
# Images are served at /images/{manual_id}/parts/... and /images/{manual_id}/subassemblies/...
if settings.cropped_dir.exists():
    app.mount(
        "/images",
        StaticFiles(directory=str(settings.cropped_dir)),
        name="images"
    )
    logger.info(f"Mounted static files at /images -> {settings.cropped_dir}")
else:
    logger.warning(f"Cropped images directory does not exist: {settings.cropped_dir}")

# Include routers
app.include_router(
    ingestion.router,
    prefix="/api/ingest",
    tags=["Ingestion"]
)
app.include_router(
    steps.router,
    prefix="/api",
    tags=["Steps"]
)
app.include_router(
    parts.router,
    prefix="/api",
    tags=["Parts"]
)
app.include_router(
    digital_twin.router,
    prefix="/api",
    tags=["Digital Twin"]
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LEGO Assembly API",
        "version": "2.0.0",
        "description": "Simplified API for LEGO manual processing",
        "endpoints": {
            "ingestion": "/api/ingest/pdf (POST)",
            "manuals": "/api/manuals (GET)",
            "steps": "/api/manuals/{manual_id}/steps (GET)",
            "parts": "/api/manuals/{manual_id}/parts (GET)",
            "images": "/images/{manual_id}/parts/... or /images/{manual_id}/subassemblies/..."
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "data_dir": str(settings.data_dir),
        "vlm_model": settings.vlm_model
    }


# Logging configuration
logger.add(
    "logs/api_{time}.log",
    rotation="1 day",
    retention="7 days",
    level=settings.log_level
)

logger.info(f"LEGO Assembly API initialized")
logger.info(f"Data directory: {settings.data_dir}")
logger.info(f"VLM model: {settings.vlm_model}")
logger.info(f"Frontend URL: {settings.frontend_url}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
