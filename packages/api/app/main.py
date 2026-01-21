"""
Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import jobs, files, logs
from .services.job_manager import job_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    await init_db()
    await job_manager.start()
    yield
    # Shutdown
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
