"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.database import init_db
from app.routes import (
    documents_router,
    extractions_router,
    vocabulary_router,
    sync_router,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting VocabExt...")

    # Ensure directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Pre-load spaCy model (optional, can be slow)
    # This makes the first request faster
    try:
        from app.services.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        tokenizer._load_model()
        logger.info("spaCy model loaded")
    except Exception as e:
        logger.warning(f"Failed to pre-load spaCy model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down VocabExt...")


# Create FastAPI app
app = FastAPI(
    title="VocabExt",
    description="German vocabulary extraction tool with Anki sync",
    version="0.1.0",
    lifespan=lifespan,
)

# Setup templates
templates_dir = Path(__file__).parent / "templates"
app.state.templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(documents_router)
app.include_router(extractions_router)
app.include_router(vocabulary_router)
app.include_router(sync_router)


@app.get("/")
async def root():
    """Redirect root to documents page."""
    return RedirectResponse(url="/documents", status_code=302)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "version": "0.1.0",
    })


def run():
    """Run the application (for use with `vocabext` command)."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
