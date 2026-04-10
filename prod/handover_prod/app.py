from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from handover_prod.config import Settings
from handover_prod.predictors import PredictionResult, PredictorRegistry
from handover_prod.schemas import (
    ChecklistPredictionResponse,
    GenericPredictionResponse,
    HealthResponse,
    PredictRequest,
    ReadyResponse,
    SpanPredictionResponse,
    TextRequest,
)

logger = logging.getLogger(__name__)


def _result_payload(result: PredictionResult) -> dict:
    return {
        "task": result.task,
        "model": result.model,
        "program_path": result.program_path,
        "latency_ms": result.latency_ms,
        "labels": result.labels,
        "pred_spans": result.pred_spans,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = PredictorRegistry.from_settings(app_settings, configure_lm=True)
        app.state.settings = app_settings
        app.state.registry = registry
        yield

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )

    def registry(request: Request) -> PredictorRegistry:
        return request.app.state.registry

    @app.get("/", response_model=ReadyResponse)
    async def root(request: Request) -> ReadyResponse:
        return ReadyResponse(status="ready", tasks=registry(request).task_metadata())

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=app_settings.app_name,
            version=app_settings.app_version,
        )

    @app.get("/readyz", response_model=ReadyResponse)
    async def readyz(request: Request) -> ReadyResponse:
        return ReadyResponse(status="ready", tasks=registry(request).task_metadata())

    async def predict_task(
        request: Request, task_name: str, text: str
    ) -> PredictionResult:
        try:
            return await run_in_threadpool(registry(request).predict, task_name, text)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown task {task_name!r}") from exc
        except Exception as exc:
            logger.exception("Prediction failed for task %s", task_name)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/predict", response_model=GenericPredictionResponse)
    async def predict(request: Request, body: PredictRequest) -> GenericPredictionResponse:
        result = await predict_task(request, body.task, body.text)
        return GenericPredictionResponse(**_result_payload(result))

    @app.post("/predict/checklist", response_model=ChecklistPredictionResponse)
    async def predict_checklist(
        request: Request, body: TextRequest
    ) -> ChecklistPredictionResponse:
        result = await predict_task(request, "checklist", body.text)
        return ChecklistPredictionResponse(**_result_payload(result))

    @app.post("/predict/sbar", response_model=SpanPredictionResponse)
    async def predict_sbar(request: Request, body: TextRequest) -> SpanPredictionResponse:
        result = await predict_task(request, "sbar", body.text)
        return SpanPredictionResponse(**_result_payload(result))

    @app.post("/predict/unknown-fact", response_model=SpanPredictionResponse)
    async def predict_unknown_fact(
        request: Request, body: TextRequest
    ) -> SpanPredictionResponse:
        result = await predict_task(request, "unknown_fact", body.text)
        return SpanPredictionResponse(**_result_payload(result))

    return app


app = create_app()
