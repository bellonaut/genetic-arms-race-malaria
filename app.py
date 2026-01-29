from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

app = FastAPI(
    title="Malaria Risk Prediction API", description="wGRS+GF+POS model deployment"
)

MODEL_PATH = Path("models/lightgbm_best.pkl")
FEATURES_PATH = Path("models/feature_names.json")


class GeneticProfile(BaseModel):
    """Accepts SNP genotype calls keyed by SNP id (e.g., rs334: 0/1/2)."""

    snps: Dict[str, int] = Field(..., description="Mapping of SNP id to genotype 0/1/2")

    @validator("snps")
    def validate_genotypes(cls, v: Dict[str, int]) -> Dict[str, int]:
        for snp, g in v.items():
            if g not in (0, 1, 2):
                raise ValueError(f"{snp} genotype must be 0, 1, or 2")
        return v


def _load_model_and_features():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise RuntimeError(
            "Model or feature metadata missing; expected models/lightgbm_best.pkl and models/feature_names.json"
        )

    model = joblib.load(MODEL_PATH)
    feature_names: List[str] = json.loads(FEATURES_PATH.read_text())
    return model, feature_names


try:
    MODEL, FEATURE_NAMES = _load_model_and_features()
except Exception:
    MODEL, FEATURE_NAMES = None, None


@app.post("/predict")
def predict_risk(profile: GeneticProfile):
    """Predict malaria risk from genetic profile."""
    if MODEL is None or FEATURE_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not available on server.")

    # Assemble feature vector in the expected order; missing SNPs default to 0
    vector = np.array([profile.snps.get(snp, 0) for snp in FEATURE_NAMES]).reshape(1, -1)

    try:
        risk_score = float(MODEL.predict(vector)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Simple interpretation threshold: median training risk ~1e-3 (tunable)
    interpretation = "high" if risk_score > 1e-3 else "low"

    return {
        "risk_score": risk_score,
        "interpretation": interpretation,
        "model_version": "tai_dhaliwal_v2.0",
        "features_used": FEATURE_NAMES,
    }
