"""
Main FastAPI application entry point.
Includes job management, file uploads, and real-time SSE streaming.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import jobs, files, logs
from .services.job_manager import job_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("=== API STARTING (v2 with job creation) ===")
    print("=== API STARTING (v2 with job creation) ===")
    await init_db()
    logger.info("Database initialized")
    await job_manager.start()
    logger.info("Job manager started")
    yield
    # Shutdown
    logger.info("=== API SHUTTING DOWN ===")
    await job_manager.stop()


app = FastAPI(
    title="Transcription Pipeline API",
    description="API for managing audio transcription jobs",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs.router, prefix="/api", tags=["jobs"])
app.include_router(files.router, prefix="/api", tags=["files"])
app.include_router(logs.router, prefix="/api", tags=["logs"])


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "ok",
        "name": "Transcription Pipeline API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
