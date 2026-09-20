from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.app.database import get_row
from api.app.services.container import report_service


router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/{experiment_id}", status_code=201)
def generate_report(experiment_id: str):
    try:
        return report_service.generate(experiment_id)
    except KeyError as exc:
        raise HTTPException(404, "Experiment not found") from exc


@router.get("/{report_id}/download")
def download_report(report_id: str):
    row = get_row("reports", report_id)
    if row is None:
        raise HTTPException(404, "Report not found")
    path = Path(row["zip_path"])
    if not path.exists():
        raise HTTPException(410, "Report file no longer exists")
    return FileResponse(path, media_type="application/zip", filename=path.name)
