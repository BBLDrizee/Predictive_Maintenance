"""TurboRUL prediction API (FR-7).

    uvicorn api.main:app --port 8000
    # docs at http://localhost:8000/docs
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from turborul import service
from turborul.data import SUBSETS

app = FastAPI(title="TurboRUL", version="0.1.0",
              description="Remaining Useful Life predictions for turbofan engines (NASA C-MAPSS, simulated data).")

Subset = Literal["FD001", "FD002", "FD003", "FD004"]
Tier = Literal["Critical", "Watch", "Healthy"]


class CycleRow(BaseModel):
    """One cycle of raw readings, as in the C-MAPSS files."""

    cycle: int = Field(ge=1)
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float


class PredictRequest(BaseModel):
    subset: Subset = Field(description="Which subset's model to use (operating conditions and fault modes differ).")
    history: list[CycleRow] = Field(min_length=1, description="The engine's history, one row per cycle.")


def _require(subset: str) -> None:
    if subset not in SUBSETS:
        raise HTTPException(404, f"unknown subset {subset}")
    if subset not in {s["subset"] for s in service.subsets()}:
        raise HTTPException(503, f"no exported model for {subset}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "models": [s["subset"] for s in service.subsets()]}


@app.get("/subsets")
def subsets() -> list[dict[str, Any]]:
    return service.subsets()


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    _require(req.subset)
    try:
        return service.predict_rows(req.subset, [r.model_dump() for r in req.history])
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/fleet/{subset}")
def fleet(subset: str, tier: Tier | None = None) -> list[dict[str, Any]]:
    _require(subset)
    return service.fleet(subset, tier)


@app.get("/fleet/{subset}/importance")
def importance(subset: str) -> list[dict[str, Any]]:
    _require(subset)
    return service.global_importance(subset)


@app.get("/engines/{subset}/{unit}")
def engine(subset: str, unit: int) -> dict[str, Any]:
    _require(subset)
    try:
        return service.engine(subset, unit)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/results")
def results() -> dict[str, Any]:
    return service.model_comparison()


@app.get("/results/{subset}/{model}/predictions")
def test_predictions(subset: str, model: Literal["linear", "xgb", "lstm", "cnn"]) -> list[dict[str, Any]]:
    try:
        return service.test_predictions(subset, model)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from e
