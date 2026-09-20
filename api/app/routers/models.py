from fastapi import APIRouter

from api.app.services.container import model_registry


router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models():
    return model_registry.list_models()
