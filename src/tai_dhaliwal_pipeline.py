#!/usr/bin/env python3
"""Tai & Dhaliwal (2022) replication validation pipeline.

Implements the wGRS+GF+POS methodology described in Tai & Dhaliwal (2022)
Applied Sciences 12(9):4513 and validation checks aligned with the
Journal of Big Data 9:85 replication notes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# ============================================================================
# 1. EXACT POPULATION SPECIFICATIONS (Table 1, Page 4)
# Total after preprocessing: 20,817 (removed 37 samples from 20,854)
# ============================================================================

POPULATIONS: Dict[str, int] = {
    "Gambia": 5593,
    "Kenya": 3682,
    "Malawi": 3088,
    "Cameroon": 1471,
    "Burkina_Faso": 1446,
    "Vietnam": 1728,
    "Tanzania": 979,
    "Mali": 869,
    "Ghana": 764,
    "PNG": 815,
    "Nigeria": 419,
}

TARGET_TOTAL = 20817


@dataclass(frozen=True)
class SNPConfig:
    """Configuration for a SNP used in Tai & Dhaliwal (2022) wGRS+GF+POS."""

    snp_id: str
    effect_size: float
    index: int


GENOTYPE_CODES = {0: 104, 1: 208, 2: 312}


def hardy_weinberg_freq(genotype: int, maf: float) -> float:
    """Compute Hardy-Weinberg genotype frequencies per Tai & Dhaliwal (2022)."""

    p = maf
    q = 1 - maf
    if genotype == 0:
        return q * q
    if genotype == 1:
        return 2 * p * q
    return p * p


def calculate_wgrs(df: pd.DataFrame, snp_config: List[SNPConfig]) -> np.ndarray:
    """Calculate wGRS+GF+POS risk score (Tai & Dhaliwal 2022 Applied Sciences 12(9):4513)."""

    scores = np.zeros(len(df), dtype=float)
    for snp in snp_config:
        genotypes = df[snp.snp_id].to_numpy()
        genotype_freq = np.vectorize(hardy_weinberg_freq)(genotypes, df[f"maf_{snp.snp_id}"].to_numpy())
        numerator = genotypes * snp.effect_size * genotype_freq
        mutation_value = np.array([GENOTYPE_CODES[g] * snp.index for g in genotypes])
        scores += numerator / mutation_value
    return scores


class MalariaDataGenerator:
    """Generate synthetic malaria genomic data following Tai & Dhaliwal (2022)."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.snp_config = self._build_snp_config()
        self.samples = self._scale_population_counts()

    def _scale_population_counts(self) -> Dict[str, int]:
        total_samples = sum(POPULATIONS.values())
        scale_factor = TARGET_TOTAL / total_samples
        scaled = {k: int(v * scale_factor) for k, v in POPULATIONS.items()}
        current_sum = sum(scaled.values())
        scaled["Nigeria"] += TARGET_TOTAL - current_sum
        return scaled

    def _build_snp_config(self) -> List[SNPConfig]:
        snps: List[SNPConfig] = []
        for idx in range(104):
            snp_id = "rs334" if idx == 0 else f"rs_{idx}"
            effect_size = float(self.rng.gamma(2, 0.3))
            if idx == 0:
                effect_size = -0.85
            snps.append(SNPConfig(snp_id=snp_id, effect_size=effect_size, index=idx + 1))
        return snps

    def generate_population_data(self, pop_name: str, n_samples: int) -> pd.DataFrame:
        """Generate Hardy-Weinberg compliant genotype data for a population."""

        n_snps = len(self.snp_config)
        maf_values = self.rng.beta(2, 5, n_snps)
        genotypes = np.zeros((n_samples, n_snps), dtype=int)
        gf_0 = (1 - maf_values) ** 2
        gf_1 = 2 * maf_values * (1 - maf_values)
        gf_2 = maf_values**2
        for idx in range(n_snps):
            genotypes[:, idx] = self.rng.choice([0, 1, 2], size=n_samples, p=[gf_0[idx], gf_1[idx], gf_2[idx]])

        df = pd.DataFrame(genotypes, columns=[snp.snp_id for snp in self.snp_config])
        for idx, snp in enumerate(self.snp_config):
            df[f"maf_{snp.snp_id}"] = maf_values[idx]
        df["Population"] = pop_name
        return df

    def generate(self) -> pd.DataFrame:
        """Generate full synthetic dataset across all populations."""

        populations = [
            self.generate_population_data(pop, count) for pop, count in self.samples.items()
        ]
        df_full = pd.concat(populations, ignore_index=True)
        df_full["wGRS_GF_POS"] = calculate_wgrs(df_full, self.snp_config)
        df_full["wGRS_GF"] = self._calculate_wgrs_gf(df_full)
        return df_full

    def _calculate_wgrs_gf(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df), dtype=float)
        for snp in self.snp_config:
            genotypes = df[snp.snp_id].to_numpy()
            genotype_freq = np.vectorize(hardy_weinberg_freq)(genotypes, df[f"maf_{snp.snp_id}"].to_numpy())
            scores += genotypes * snp.effect_size * genotype_freq
        return scores


class MalariaRiskModels:
    """Train malaria risk prediction models."""

    def __init__(self) -> None:
        self.models = {
            "LightGBM": LGBMRegressor(
                objective="regression",
                metric="mae",
                num_leaves=31,
                max_depth=8,
                learning_rate=0.05,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1,
                random_state=42,
            ),
            "Ridge": Ridge(alpha=10),
            "SVR": SVR(kernel="rbf", C=1.0),
            "MLP_8-8-8-1": MLPRegressor(
                hidden_layer_sizes=(8, 8, 8),
                activation="elu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=100,
                random_state=42,
            ),
        }

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, X_test: np.ndarray) -> Dict[str, float]:
        """Train all models and return MAE on test data."""

        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = float(mean_absolute_error(y_test, preds))
        return results


def fairness_audit(predictions: np.ndarray, y_true: np.ndarray, groups: Iterable[str]) -> Dict[str, object]:
    """Compute IEEE 2857-2021 fairness metrics."""

    groups = np.asarray(list(groups))
    y_true = np.asarray(y_true)
    preds = np.asarray(predictions)
    y_pred = (preds > np.median(preds)).astype(int)

    tpr_by_group: Dict[str, float] = {}
    calibration_errors: List[float] = []

    for group in np.unique(groups):
        mask = groups == group
        positives = np.sum(y_true[mask] == 1)
        true_positive = np.sum((y_pred[mask] == 1) & (y_true[mask] == 1))
        tpr = float(true_positive / positives) if positives > 0 else 0.0
        tpr_by_group[str(group)] = tpr
        calibration_errors.append(float(np.mean(preds[mask]) - np.mean(y_true[mask])))

    tpr_values = list(tpr_by_group.values())
    return {
        "overall_tpr_disparity": float(max(tpr_values) - min(tpr_values)) if tpr_values else 0.0,
        "max_calibration_error": float(max(abs(val) for val in calibration_errors))
        if calibration_errors
        else 0.0,
        "population_tpr": tpr_by_group,
    }


def population_distribution(samples: Dict[str, int]) -> pd.DataFrame:
    """Build population distribution table."""

    rows = []
    for pop, count in samples.items():
        rows.append(
            {
                "Population": pop,
                "Count": count,
                "Percentage": round((count / TARGET_TOTAL) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run full data pipeline and export outputs."""

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MalariaDataGenerator(random_state=42)
    df_full = generator.generate()

    print("=" * 60)
    print("POPULATION VALIDATION (Tai & Dhaliwal 2022, Table 1)")
    print("=" * 60)
    for pop, count in generator.samples.items():
        pct = 100 * count / TARGET_TOTAL
        print(f"{pop:15s}: {count:5d} ({pct:.2f}%)")
    print(f"{'TOTAL':15s}: {sum(generator.samples.values()):5d}")
    print()

    print(f"rs334 protective effect direction: {np.corrcoef(df_full['rs334'], df_full['wGRS_GF_POS'])[0,1]:.3f}")

    sample_idx = 0
    snp_idx = 0
    g = int(df_full.iloc[sample_idx][generator.snp_config[snp_idx].snp_id])
    mutation_val = GENOTYPE_CODES[g] * (snp_idx + 1)
    print("Formula Check (Sample 0, SNP 0 [rs334]):")
    print(f"  Genotype: {g} (code: {GENOTYPE_CODES[g]})")
    print(f"  Column index: {snp_idx + 1}")
    print(f"  Mutation value: {mutation_val}")

    population_distribution(generator.samples).to_csv(
        output_dir / "population_distribution.csv", index=False
    )
    df_full.to_csv(output_dir / "synthetic_clinical.csv", index=False)

    feature_cols = [snp.snp_id for snp in generator.snp_config]
    X = df_full[feature_cols].to_numpy()
    y_pos = df_full["wGRS_GF_POS"].to_numpy()

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y_pos,
        df_full["Population"].to_numpy(),
        test_size=0.15,
        random_state=42,
        stratify=df_full["Population"].to_numpy(),
    )

    models = MalariaRiskModels()
    results_pos = models.train_all(X_train, y_train, y_test, X_test)

    with open(output_dir / "model_metrics.json", "w") as handle:
        json.dump(results_pos, handle, indent=2)

    lgbm_preds = models.models["LightGBM"].predict(X_test)
    fairness = fairness_audit(lgbm_preds, (y_test > np.median(y_test)).astype(int), groups_test)

    with open(output_dir / "fairness_metrics.json", "w") as handle:
        json.dump(fairness, handle, indent=2)


if __name__ == "__main__":
    main()
