from fastapi import APIRouter, HTTPException

from api.app.schemas.models import ComparisonCreate, SimulationSpeedUpdate
from api.app.services.container import comparison_service


router = APIRouter(prefix="/api/comparisons", tags=["comparisons"])


def _handle(action):
    try:
        return action()
    except KeyError as exc:
        raise HTTPException(404, "Comparison not found") from exc
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("", status_code=202)
def start_comparison(request: ComparisonCreate):
    return _handle(lambda: comparison_service.start(request))


@router.get("/{experiment_id}")
def get_comparison(experiment_id: str):
    return _handle(lambda: comparison_service.get(experiment_id))


@router.post("/{experiment_id}/pause")
def pause_comparison(experiment_id: str):
    return _handle(lambda: comparison_service.pause(experiment_id))


@router.post("/{experiment_id}/resume")
def resume_comparison(experiment_id: str):
    return _handle(lambda: comparison_service.resume(experiment_id))


@router.post("/{experiment_id}/stop")
def stop_comparison(experiment_id: str):
    return _handle(lambda: comparison_service.stop(experiment_id))


@router.post("/{experiment_id}/speed")
def set_comparison_speed(experiment_id: str, request: SimulationSpeedUpdate):
    return _handle(lambda: comparison_service.set_speed(experiment_id, request.speed))
