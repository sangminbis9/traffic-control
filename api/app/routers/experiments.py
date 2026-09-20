import json

from fastapi import APIRouter, HTTPException

from api.app.database import get_row, list_rows


router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _decode(row):
    if row is None:
        return None
    output = dict(row)
    for key in ("config_json", "result_json", "state_json", "metadata_json"):
        if key in output:
            output[key.removesuffix("_json")] = json.loads(output.pop(key))
    return output


@router.get("")
def list_experiments():
    return [_decode(row) for row in list_rows("experiments")]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str):
    row = _decode(get_row("experiments", experiment_id))
    if row is None:
        raise HTTPException(404, "Experiment not found")
    return row
