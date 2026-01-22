"""
Resources Router - Exposes GPU and RAM status for dashboard.
"""

from fastapi import APIRouter

from ..services.job_manager import job_manager

router = APIRouter()


@router.get("/resources")
async def get_resources():
    """
    Get current resource utilization status.

    Returns GPU slot usage and system RAM availability.
    """
    return job_manager.get_resource_status()
