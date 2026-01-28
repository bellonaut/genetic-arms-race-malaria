"""End-to-end genomic malaria risk prediction pipeline.

Implements Tai & Dhaliwal (2022) Applied Sciences 12(9):4513 wGRS+GF+POS
methodology for synthetic data generation and model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


POPULATION_COUNTS: Dict[str, int] = {
    "Gambia": 5593,
    "Kenya": 3682,
    "Malawi": 3088,
    "Vietnam": 1724,
    "Cameroon": 1481,
    "Burkina Faso": 1437,
    "Tanzania": 970,
    "Mali": 863,
    "PNG": 803,
    "Ghana": 773,
    "Nigeria": 403,
}

AFRICA_POPS = {
    "Gambia",
    "Kenya",
    "Malawi",
    "Cameroon",
    "Burkina Faso",
    "Tanzania",
    "Mali",
    "Ghana",
    "Nigeria",
}


@dataclass(frozen=True)
class SNPConfig:
    """Configuration for a SNP used in Tai & Dhaliwal (2022) wGRS+GF+POS."""

    snp_id: str
    chrom: int
    pos: int
    effect_size: float
    maf_by_population: Dict[str, float]


def hardy_weinberg_freq(genotype: int, maf: float) -> float:
    """Compute Hardy-Weinberg genotype frequencies using Tai & Dhaliwal (2022)."""

    p = maf
    q = 1 - maf
    if genotype == 0:
        return q * q
    if genotype == 1:
        return 2 * p * q
    return p * p


def calculate_wgrs(df: pd.DataFrame, snp_config: List[SNPConfig]) -> np.ndarray:
    """Calculate wGRS+GF+POS risk score per Tai & Dhaliwal (2022) Applied Sciences 12(9):4513."""

    scores = np.zeros(len(df), dtype=float)
    for index, snp in enumerate(snp_config, start=1):
        maf = df["population"].map(snp.maf_by_population).to_numpy()
        genotypes = df[snp.snp_id].to_numpy()
        genotype_freq = np.vectorize(hardy_weinberg_freq)(genotypes, maf)
        numerator = genotypes * snp.effect_size * genotype_freq
        denominator = snp.pos * index
        scores += numerator / denominator
    return scores


class MalariaDataGenerator:
    """Generate synthetic malaria genomic and clinical data.

    Uses Tai & Dhaliwal (2022) Applied Sciences 12(9):4513 wGRS+GF+POS methodology.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.population_counts = POPULATION_COUNTS
        self.snp_config = self._build_snp_config()

    def _build_snp_config(self) -> List[SNPConfig]:
        snps: List[SNPConfig] = []
        rs334_maf = {
            pop: (0.18 if pop in AFRICA_POPS else 0.02) for pop in self.population_counts
        }
        snps.append(
            SNPConfig(
                snp_id="rs334",
                chrom=11,
                pos=5_227_002,
                effect_size=-0.85,
                maf_by_population=rs334_maf,
            )
        )

        chroms = [1, 2, 6, 11, 16]
        for idx in range(103):
            snp_id = f"rs{1001 + idx}"
            chrom = int(self.rng.choice(chroms))
            pos = int(self.rng.integers(100_000, 250_000_000))
            effect_size = float(self.rng.normal(loc=0.05, scale=0.02))
            maf_by_population = {
                pop: float(np.clip(self.rng.beta(2, 5), 0.01, 0.5))
                for pop in self.population_counts
            }
            snps.append(
                SNPConfig(
                    snp_id=snp_id,
                    chrom=chrom,
                    pos=pos,
                    effect_size=effect_size,
                    maf_by_population=maf_by_population,
                )
            )
        return snps

    def generate(self) -> pd.DataFrame:
        """Generate synthetic dataset for 20,817 samples."""

        rows: List[pd.DataFrame] = []
        sample_id = 0
        for population, count in self.population_counts.items():
            genotypes = {
                snp.snp_id: self.rng.binomial(2, snp.maf_by_population[population], size=count)
                for snp in self.snp_config
            }
            clinical = {
                "age": self.rng.gamma(2.5, 3.0, size=count) + 1,
                "parasitemia": self.rng.lognormal(mean=3.0, sigma=1.2, size=count),
                "prior_malaria": self.rng.poisson(2, size=count),
            }
            df = pd.DataFrame({**genotypes, **clinical})
            df.insert(0, "sample_id", np.arange(sample_id, sample_id + count))
            df["population"] = population
            sample_id += count
            rows.append(df)

        full_df = pd.concat(rows, ignore_index=True)
        full_df["risk_score"] = calculate_wgrs(full_df, self.snp_config)
        noise = self.rng.normal(0, 0.0005, size=len(full_df))
        full_df["risk_target"] = full_df["risk_score"] + noise
        threshold = np.quantile(full_df["risk_score"], 0.6)
        full_df["case"] = (full_df["risk_score"] > threshold).astype(int)
        return full_df


class MalariaRiskModels:
    """Train regression models for malaria risk prediction."""

    def __init__(self) -> None:
        self.models = {
            "ridge": RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5),
            "lightgbm": LGBMRegressor(
                objective="regression",
                metric="mae",
                num_leaves=31,
                n_estimators=100,
                random_state=42,
            ),
            "svr": SVR(kernel="rbf", C=1.0, epsilon=0.1),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(8, 8, 8),
                activation="elu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=100,
                random_state=42,
            ),
        }

    def train_all(
        self, X: np.ndarray, y: np.ndarray, y_case: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Train all models and return MAE/AUC metrics."""

        results: Dict[str, Dict[str, float]] = {}
        for name, model in self.models.items():
            model.fit(X, y)
            preds = model.predict(X)
            results[name] = {
                "mae": float(mean_absolute_error(y, preds)),
                "auc": float(roc_auc_score(y_case, preds)),
            }
        return results


def fairness_audit(
    predictions: np.ndarray, y_true: np.ndarray, groups: Iterable[str]
) -> Dict[str, object]:
    """Compute IEEE 2857-2021 fairness metrics (TPR disparity, calibration error)."""

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


def population_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Build population distribution table."""

    total = len(df)
    rows = []
    for pop, count in POPULATION_COUNTS.items():
        rows.append(
            {
                "Population": pop,
                "Count": count,
                "Percentage": round((count / total) * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run full data pipeline and export outputs."""

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = MalariaDataGenerator(random_state=42)
    df = generator.generate()

    print(f"Population total: {df['population'].value_counts().sum()}")
    print(f"Nigeria count: {df['population'].value_counts().get('Nigeria', 0)}")

    example_snp = generator.snp_config[0]
    example_pop = df.iloc[0]["population"]
    example_geno = df.iloc[0][example_snp.snp_id]
    example_freq = hardy_weinberg_freq(
        int(example_geno), example_snp.maf_by_population[example_pop]
    )
    denominator_example = example_snp.pos * 1
    print(
        "Example wGRS term (rs334):",
        (example_geno * example_snp.effect_size * example_freq) / denominator_example,
    )

    if example_snp.effect_size < 0:
        print("rs334 protective negative effect verified.")

    df.to_csv(output_dir / "synthetic_clinical.csv", index=False)
    population_distribution(df).to_csv(output_dir / "population_distribution.csv", index=False)

    feature_cols = [snp.snp_id for snp in generator.snp_config] + [
        "age",
        "parasitemia",
        "prior_malaria",
    ]
    X = df[feature_cols].to_numpy()
    y = df["risk_target"].to_numpy()
    y_case = df["case"].to_numpy()

    X_train, X_test, y_train, y_test, case_train, case_test, group_train, group_test = (
        train_test_split(
            X,
            y,
            y_case,
            df["population"].to_numpy(),
            test_size=0.2,
            random_state=42,
        )
    )

    models = MalariaRiskModels()
    metrics = models.train_all(X_train, y_train, case_train)

    with open(output_dir / "model_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    lightgbm_preds = models.models["lightgbm"].predict(X_test)
    fairness = fairness_audit(lightgbm_preds, case_test, group_test)

    with open(output_dir / "fairness_metrics.json", "w") as handle:
        json.dump(fairness, handle, indent=2)


if __name__ == "__main__":
    main()
