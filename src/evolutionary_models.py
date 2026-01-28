"""Evolutionary modeling utilities for sickle-cell malaria co-evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import lightgbm as lgb


@dataclass(frozen=True)
class SelectionScenario:
    """Container for selection coefficients in a host-parasite scenario.

    Attributes:
        parasite_prevalence: Fraction of infections due to Pfsa+ parasites.
        s_hbaas_pfsa_plus: Selection coefficient for HbAS under Pfsa+.
        s_hbaas_pfsa_minus: Selection coefficient for HbAS under Pfsa-.
        s_hbaa_pfsa_plus: Selection coefficient for HbAA under Pfsa+.
        s_hbaa_pfsa_minus: Selection coefficient for HbAA under Pfsa-.
    """

    parasite_prevalence: float
    s_hbaas_pfsa_plus: float
    s_hbaas_pfsa_minus: float
    s_hbaa_pfsa_plus: float
    s_hbaa_pfsa_minus: float


def heterozygote_advantage_frequency(
    fitness_hbaas: float, fitness_hbaa: float, fitness_hbbss: float
) -> float:
    """Compute equilibrium frequency under heterozygote advantage.

    Args:
        fitness_hbaas: Relative fitness of HbAS heterozygotes.
        fitness_hbaa: Relative fitness of HbAA homozygotes.
        fitness_hbbss: Relative fitness of HbSS homozygotes.

    Returns:
        Equilibrium HbS allele frequency under overdominance.
    """

    numerator = fitness_hbaas - fitness_hbaa
    denominator = (fitness_hbaas - fitness_hbaa) + (fitness_hbaas - fitness_hbbss)
    if denominator == 0:
        raise ValueError("Denominator is zero; check fitness values.")
    return numerator / denominator


def selection_coefficients(
    parasite_prevalence: float,
    rr_pfsa_plus: float,
    rr_pfsa_minus: float,
) -> SelectionScenario:
    """Estimate selection coefficients for HbAS vs HbAA by parasite prevalence.

    Args:
        parasite_prevalence: Fraction of Pfsa+ infections in the population.
        rr_pfsa_plus: Relative risk of severe malaria for HbAS vs HbAA under Pfsa+.
        rr_pfsa_minus: Relative risk of severe malaria for HbAS vs HbAA under Pfsa-.

    Returns:
        SelectionScenario with derived coefficients for HbAS and HbAA.
    """

    if not 0 <= parasite_prevalence <= 1:
        raise ValueError("parasite_prevalence must be between 0 and 1.")

    s_hbaas_pfsa_plus = 1 - rr_pfsa_plus
    s_hbaas_pfsa_minus = 1 - rr_pfsa_minus
    s_hbaa_pfsa_plus = 0.0
    s_hbaa_pfsa_minus = 0.0

    return SelectionScenario(
        parasite_prevalence=parasite_prevalence,
        s_hbaas_pfsa_plus=s_hbaas_pfsa_plus,
        s_hbaas_pfsa_minus=s_hbaas_pfsa_minus,
        s_hbaa_pfsa_plus=s_hbaa_pfsa_plus,
        s_hbaa_pfsa_minus=s_hbaa_pfsa_minus,
    )


def weighted_selection(scenario: SelectionScenario) -> float:
    """Compute weighted selection coefficient for HbAS.

    Args:
        scenario: SelectionScenario instance.

    Returns:
        Weighted selection coefficient for HbAS across parasite genotypes.
    """

    p = scenario.parasite_prevalence
    return (p * scenario.s_hbaas_pfsa_plus) + ((1 - p) * scenario.s_hbaas_pfsa_minus)


def manhattan_plot(
    positions: Iterable[int],
    p_values: Iterable[float],
    chromosomes: Iterable[int],
    title: str,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a Manhattan-style plot from mock GWAS data.

    Args:
        positions: Genomic positions (base pairs).
        p_values: P-values for each locus.
        chromosomes: Chromosome identifiers.
        title: Plot title.

    Returns:
        Tuple of matplotlib figure and axes.
    """

    positions = np.asarray(list(positions))
    p_values = np.asarray(list(p_values))
    chromosomes = np.asarray(list(chromosomes))

    if not (len(positions) == len(p_values) == len(chromosomes)):
        raise ValueError("positions, p_values, and chromosomes must be the same length.")

    minus_log_p = -np.log10(p_values)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = np.where(chromosomes % 2 == 0, "#2c7fb8", "#7fcdbb")
    ax.scatter(positions, minus_log_p, c=colors, s=18, alpha=0.8, edgecolor="none")
    ax.set_xlabel("Genomic position (bp)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


class MalariaRiskPredictor:
    """Multi-model malaria risk prediction with fairness auditing."""

    def __init__(self) -> None:
        """Initialize empty model registry and scalers."""
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_cols: List[str] = []

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix using SNPs and clinical covariates.

        Args:
            df: DataFrame with SNP columns and clinical features.

        Returns:
            Tuple of feature matrix, labels, and feature names.
        """
        snp_cols = [col for col in df.columns if col.startswith("rs")]
        clinical_cols = ["age", "parasitemia", "prior_malaria"]
        self.feature_cols = snp_cols + clinical_cols

        X = df[self.feature_cols].fillna(0).to_numpy()
        y = df["case"].fillna(0).to_numpy()
        return X, y, self.feature_cols

    def train_ridge(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Ridge Regression with hyperparameter tuning.

        Args:
            X: Feature matrix.
            y: Binary labels.

        Returns:
            Dictionary with model, best alpha, and validation MAE.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}
        ridge = Ridge()
        search = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_absolute_error")
        search.fit(X_scaled, y)

        self.models["ridge"] = search.best_estimator_
        self.scalers["ridge"] = scaler

        return {
            "model": search.best_estimator_,
            "best_alpha": float(search.best_params_["alpha"]),
            "mae": float(-search.best_score_),
            "cv_results": search.cv_results_,
        }

    def train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM with cross-validation.

        Args:
            X: Feature matrix.
            y: Binary labels.

        Returns:
            Dictionary with model, MAE, and feature importance.
        """
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores: List[float] = []
        model = None

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(10, verbose=False)],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            scores.append(mean_absolute_error(y_val, preds))

        if model is None:
            raise RuntimeError("LightGBM training failed to produce a model.")

        self.models["lightgbm"] = model

        return {
            "model": model,
            "mae": float(np.mean(scores)),
            "feature_importance": model.feature_importance(importance_type="gain"),
        }

    def train_svr(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Support Vector Regression.

        Args:
            X: Feature matrix.
            y: Binary labels.

        Returns:
            Dictionary with model and scaler.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        svr = SVR(kernel="rbf", C=1.0)
        svr.fit(X_scaled, y)

        self.models["svr"] = svr
        self.scalers["svr"] = scaler

        return {"model": svr}

    def fairness_metrics(
        self, X: np.ndarray, y_true: np.ndarray, groups: np.ndarray, model_name: str
    ) -> Dict[str, Any]:
        """Calculate fairness metrics across population groups.

        Metrics include TPR/FPR disparities and calibration error by population.

        Args:
            X: Feature matrix.
            y_true: True binary labels.
            groups: Population group labels for each sample.
            model_name: Name of the trained model to evaluate.

        Returns:
            Dictionary containing per-group metrics and disparities.
        """
        model = self.models[model_name]

        if model_name in self.scalers:
            X_in = self.scalers[model_name].transform(X)
            y_pred_proba = model.predict(X_in)
        else:
            y_pred_proba = model.predict(X)

        y_pred = (y_pred_proba > 0.5).astype(int)
        unique_groups = np.unique(groups)

        results: Dict[str, Dict[str, float]] = {}
        for group in unique_groups:
            mask = groups == group
            y_group = y_true[mask]
            y_pred_group = y_pred[mask]
            y_proba_group = y_pred_proba[mask]

            positives = np.sum(y_group == 1)
            negatives = np.sum(y_group == 0)
            tpr = (
                float(np.sum((y_pred_group == 1) & (y_group == 1)) / positives)
                if positives > 0
                else 0.0
            )
            fpr = (
                float(np.sum((y_pred_group == 1) & (y_group == 0)) / negatives)
                if negatives > 0
                else 0.0
            )
            positive_rate = float(np.mean(y_pred_group)) if len(y_pred_group) > 0 else 0.0
            calibration_error = float(np.mean(y_proba_group) - np.mean(y_group))

            results[str(group)] = {
                "tpr": tpr,
                "fpr": fpr,
                "positive_rate": positive_rate,
                "calibration_error": calibration_error,
                "n_samples": float(len(y_group)),
            }

        tprs = [results[group]["tpr"] for group in results]
        fprs = [results[group]["fpr"] for group in results]
        calibrations = [abs(results[group]["calibration_error"]) for group in results]

        return {
            "by_group": results,
            "tpr_disparity": float(max(tprs) - min(tprs)) if tprs else 0.0,
            "fpr_disparity": float(max(fprs) - min(fprs)) if fprs else 0.0,
            "max_calibration_error": float(max(calibrations)) if calibrations else 0.0,
        }
