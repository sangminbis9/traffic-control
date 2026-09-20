"""Traffic Control Lab FastAPI entry point."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from queue import Queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.app.database import initialize_database
from api.app.routers import comparisons, experiments, models, reports, training
from api.app.services.container import comparison_service, training_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="Traffic Control Lab API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for router in (models.router, training.router, comparisons.router, experiments.router, reports.router):
    app.include_router(router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


async def _stream(
    websocket: WebSocket,
    session_id: str,
    hub,
    initial_factory: Callable[[], dict],
    on_disconnect=None,
):
    await websocket.accept()
    queue: Queue = hub.subscribe(session_id)
    try:
        # Subscribe before taking the initial snapshot so a fast runner cannot
        # finish in the gap between the snapshot and event subscription.
        await websocket.send_json(initial_factory())
        while True:
            try:
                event = await asyncio.wait_for(asyncio.to_thread(queue.get), timeout=5.0)
                await websocket.send_json(event)
            except TimeoutError:
                await websocket.send_json({"type": "heartbeat", "session_id": session_id})
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(session_id, queue)
        if on_disconnect is not None:
            try:
                on_disconnect()
            except (KeyError, RuntimeError):
                pass


@app.websocket("/ws/training/{session_id}")
async def training_socket(websocket: WebSocket, session_id: str):
    try:
        training_service.get(session_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await _stream(
        websocket,
        session_id,
        training_service.events,
        lambda: training_service.get(session_id),
    )


@app.websocket("/ws/comparisons/{experiment_id}")
async def comparison_socket(websocket: WebSocket, experiment_id: str):
    try:
        comparison_service.get(experiment_id)
    except KeyError:
        await websocket.close(code=4404)
        return
    await _stream(
        websocket,
        experiment_id,
        comparison_service.events,
        lambda: comparison_service.get(experiment_id),
        on_disconnect=lambda: comparison_service.stop(experiment_id),
    )


def main() -> None:
    import uvicorn

    uvicorn.run("api.app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
