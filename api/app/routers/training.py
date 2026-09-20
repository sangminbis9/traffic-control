from fastapi import APIRouter, HTTPException

from api.app.schemas.models import TrainingCreate
from api.app.services.container import training_service


router = APIRouter(prefix="/api/training", tags=["training"])


def _handle(action):
    try:
        return action()
    except KeyError as exc:
        raise HTTPException(404, "Training session not found") from exc
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("", status_code=202)
def start_training(request: TrainingCreate):
    return training_service.start(request)


@router.get("")
def list_training():
    return training_service.list()


@router.get("/{session_id}")
def get_training(session_id: str):
    return _handle(lambda: training_service.get(session_id))


@router.post("/{session_id}/pause")
def pause_training(session_id: str):
    return _handle(lambda: training_service.pause(session_id))


@router.post("/{session_id}/resume")
def resume_training(session_id: str):
    return _handle(lambda: training_service.resume(session_id))


@router.post("/{session_id}/stop")
def stop_training(session_id: str):
    return _handle(lambda: training_service.stop(session_id))
