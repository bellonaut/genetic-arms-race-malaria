"""Evolutionary modeling utilities for sickle-cell malaria co-evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC using the rank method."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    sum_ranks_pos = np.sum(ranks[pos])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


class StandardScaler:
    """Minimal standard scaler implementation."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


class RidgeModel:
    """Closed-form ridge regression."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        X_aug = np.c_[np.ones(X.shape[0]), X]
        identity = np.eye(X_aug.shape[1])
        identity[0, 0] = 0
        weights = np.linalg.solve(X_aug.T @ X_aug + self.alpha * identity, X_aug.T @ y)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model has not been fitted.")
        return X @ self.coef_ + self.intercept_


class RFFSVRModel:
    """Random Fourier feature approximation to an RBF SVR."""

    def __init__(self, C: float = 1.0, gamma: float = 0.1, n_components: int = 200) -> None:
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        self.random_weights: np.ndarray | None = None
        self.random_offset: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.random_weights is None or self.random_offset is None:
            raise ValueError("Model has not been initialized.")
        projection = X @ self.random_weights + self.random_offset
        return np.sqrt(2 / self.n_components) * np.cos(projection)

    def fit(self, X: np.ndarray, y: np.ndarray, random_state: int = 42) -> "RFFSVRModel":
        rng = np.random.default_rng(random_state)
        self.random_weights = rng.normal(scale=np.sqrt(2 * self.gamma), size=(X.shape[1], self.n_components))
        self.random_offset = rng.uniform(0, 2 * np.pi, size=self.n_components)
        Z = self._transform(X)
        X_aug = np.c_[np.ones(Z.shape[0]), Z]
        identity = np.eye(X_aug.shape[1])
        identity[0, 0] = 0
        alpha = 1 / self.C
        weights = np.linalg.solve(X_aug.T @ X_aug + alpha * identity, X_aug.T @ y)
        self.intercept_ = float(weights[0])
        self.coef_ = weights[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model has not been fitted.")
        Z = self._transform(X)
        return Z @ self.coef_ + self.intercept_


class LightGBMModel:
    """Lightweight gradient descent logistic model with LightGBM-like API."""

    def __init__(self, learning_rate: float = 0.05, n_iter: int = 200) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMModel":
        rng = np.random.default_rng(42)
        self.coef_ = rng.normal(scale=0.01, size=X.shape[1])
        self.intercept_ = 0.0
        for _ in range(self.n_iter):
            logits = X @ self.coef_ + self.intercept_
            preds = _sigmoid(logits)
            error = preds - y
            grad_w = X.T @ error / len(y)
            grad_b = np.mean(error)
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
        return self

    def predict(self, X: np.ndarray, pred_contrib: bool = False) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted.")
        if pred_contrib:
            contribs = X * self.coef_
            base = np.full((X.shape[0], 1), self.intercept_)
            return np.hstack([contribs, base])
        return _sigmoid(X @ self.coef_ + self.intercept_)

    def feature_importance(self) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model has not been fitted.")
        return np.abs(self.coef_)


def _stratified_kfold(y: np.ndarray, n_splits: int, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    indices = np.arange(len(y))
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        label_idx = indices[y == label]
        rng.shuffle(label_idx)
        for i, idx in enumerate(label_idx):
            folds[i % n_splits].append(int(idx))
    split_indices = []
    for i in range(n_splits):
        val_idx = np.array(folds[i], dtype=int)
        train_idx = np.setdiff1d(indices, val_idx)
        split_indices.append((train_idx, val_idx))
    return split_indices

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
        clinical_cols = ["age", "parasitemia", "prior_malaria", "risk_score"]
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

        param_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
        best_alpha = param_grid[0]
        best_score = float("inf")

        for alpha in param_grid:
            scores = []
            for train_idx, val_idx in _stratified_kfold(y, n_splits=5):
                model = RidgeModel(alpha=alpha).fit(X_scaled[train_idx], y[train_idx])
                preds = model.predict(X_scaled[val_idx])
                scores.append(mean_absolute_error(y[val_idx], preds))
            avg_score = float(np.mean(scores))
            if avg_score < best_score:
                best_score = avg_score
                best_alpha = alpha

        best_model = RidgeModel(alpha=best_alpha).fit(X_scaled, y)
        self.models["ridge"] = best_model
        self.scalers["ridge"] = scaler

        return {
            "model": best_model,
            "best_alpha": float(best_alpha),
            "mae": float(best_score),
        }

    def train_lightgbm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LightGBM with cross-validation.

        Args:
            X: Feature matrix.
            y: Binary labels.

        Returns:
            Dictionary with model, MAE, and feature importance.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        scores: List[float] = []
        model = LightGBMModel(learning_rate=0.2, n_iter=2000)

        for train_idx, val_idx in _stratified_kfold(y, n_splits=5):
            model_fold = LightGBMModel(learning_rate=0.2, n_iter=2000)
            model_fold.fit(X_scaled[train_idx], y[train_idx])
            preds = model_fold.predict(X_scaled[val_idx])
            scores.append(mean_absolute_error(y[val_idx], preds))

        model.fit(X_scaled, y)
        self.models["lightgbm"] = model
        self.scalers["lightgbm"] = scaler

        return {
            "model": model,
            "mae": float(np.mean(scores)),
            "feature_importance": model.feature_importance(),
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

        svr = RFFSVRModel(C=1.0, gamma=0.1, n_components=200)
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

        y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)
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
