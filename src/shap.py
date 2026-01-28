"""Lightweight SHAP-style utilities for LightGBM models.

This module provides a minimal subset of the SHAP API needed by the slide pipeline
when the full SHAP package is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Explanation:
    values: np.ndarray
    data: np.ndarray
    feature_names: Optional[List[str]] = None
    base_values: Optional[np.ndarray] = None

    def __getitem__(self, index: int) -> "Explanation":
        return Explanation(
            values=self.values[index],
            data=self.data[index],
            feature_names=self.feature_names,
            base_values=self.base_values[index]
            if self.base_values is not None
            else None,
        )


class TreeExplainer:
    """Minimal TreeExplainer wrapper for LightGBM models."""

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, data: np.ndarray, feature_names: Optional[List[str]] = None) -> Explanation:
        contributions = self.model.predict(data, pred_contrib=True)
        values = contributions[:, :-1]
        base_values = contributions[:, -1]
        return Explanation(values=values, data=data, feature_names=feature_names, base_values=base_values)


def sample(data: np.ndarray, n: int, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(data) <= n:
        return data
    indices = rng.choice(len(data), size=n, replace=False)
    return data[indices]


def summary_plot(
    shap_values: Explanation,
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display: int = 10,
    show: bool = True,
) -> None:
    values = shap_values.values
    feature_names = feature_names or shap_values.feature_names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(values.shape[1])]

    mean_abs = np.mean(np.abs(values), axis=0)
    top_idx = np.argsort(mean_abs)[-max_display:][::-1]

    fig, ax = plt.subplots()
    cmap = plt.cm.coolwarm
    for pos, idx in enumerate(top_idx):
        shap_vals = values[:, idx]
        feat_vals = features[:, idx]
        colors = cmap((feat_vals - feat_vals.min()) / (np.ptp(feat_vals) + 1e-6))
        y = np.full_like(shap_vals, pos, dtype=float)
        ax.scatter(shap_vals, y, c=colors, s=8, alpha=0.7, edgecolors="none")

    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[idx] for idx in top_idx])
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP Summary")

    if show:
        plt.show()


def _waterfall_order(values: np.ndarray) -> Iterable[int]:
    return np.argsort(np.abs(values))[::-1]


def waterfall(shap_values: Explanation, max_display: int = 10, show: bool = True) -> None:
    values = shap_values.values
    base = shap_values.base_values if shap_values.base_values is not None else 0.0
    data = shap_values.data
    feature_names = shap_values.feature_names or [f"feature_{i}" for i in range(len(values))]

    order = list(_waterfall_order(values))[:max_display]
    contributions = values[order]
    labels = [feature_names[i] for i in order]

    cumulative = base + np.cumsum(contributions)
    fig, ax = plt.subplots()
    ax.barh(labels, contributions, color=["#d95f02" if val > 0 else "#1b9e77" for val in contributions])
    ax.axvline(base, color="black", linestyle="--", linewidth=1, label="Base value")
    ax.set_xlabel("Contribution to prediction")
    ax.set_title("SHAP Waterfall")
    ax.invert_yaxis()

    if show:
        plt.show()


class _Plots:
    def waterfall(self, shap_values: Explanation, max_display: int = 10, show: bool = True) -> None:
        waterfall(shap_values, max_display=max_display, show=show)


plots = _Plots()
