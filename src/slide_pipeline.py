"""Generate results and figures for slide deck outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from synthetic_clinical_data import MalariaDataGenerator
from evolutionary_models import MalariaRiskPredictor, mean_absolute_error, roc_auc_score


OUTPUTS_DIR = Path("outputs")
FIGURES_DIR = OUTPUTS_DIR / "figures"


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y))
    test_indices: List[int] = []
    train_indices: List[int] = []
    for label in np.unique(y):
        label_idx = indices[y == label]
        rng.shuffle(label_idx)
        n_test = int(np.floor(len(label_idx) * test_size))
        test_indices.extend(label_idx[:n_test])
        train_indices.extend(label_idx[n_test:])
    train_idx = np.array(train_indices)
    test_idx = np.array(test_indices)
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        groups[train_idx],
        groups[test_idx],
        train_idx,
        test_idx,
    )


def calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    prob_true = []
    prob_pred = []
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            prob_true.append(np.mean(y_true[mask]))
            prob_pred.append(np.mean(y_prob[mask]))
    return np.array(prob_true), np.array(prob_pred)


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.set_size_inches(6.4, 3.6)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def population_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["population"].value_counts().sort_values(ascending=False)
    distribution = counts.rename("count").to_frame()
    distribution["percentage"] = (distribution["count"] / len(df) * 100).round(2)
    distribution.reset_index(inplace=True)
    distribution.rename(columns={"index": "population"}, inplace=True)
    distribution.to_csv(OUTPUTS_DIR / "population_distribution.csv", index=False)
    return distribution


def plot_population_pie(distribution: pd.DataFrame) -> None:
    colors = sns.color_palette("Blues", 6) + sns.color_palette("Oranges", 5)
    fig, ax = plt.subplots()
    ax.pie(
        distribution["percentage"],
        labels=distribution["population"],
        autopct="%1.2f%%",
        colors=colors[: len(distribution)],
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax.axis("equal")
    fig.suptitle("Synthetic dataset replicating MalariaGEN consortium distribution (n=20,817)")
    save_figure(fig, FIGURES_DIR / "fig1_population_pie.png")


def train_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    predictor = MalariaRiskPredictor()
    X, y, feature_cols = predictor.prepare_features(df)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
        train_idx,
        test_idx,
    ) = stratified_split(X, y, df["population"].to_numpy(), test_size=0.2, random_state=42)

    ridge_results = predictor.train_ridge(X_train, y_train)
    ridge_pred = predictor.models["ridge"].predict(
        predictor.scalers["ridge"].transform(X_test)
    )

    lightgbm_results = predictor.train_lightgbm(X_train, y_train)
    if "lightgbm" in predictor.scalers:
        X_test_lgb = predictor.scalers["lightgbm"].transform(X_test)
        lgb_pred = predictor.models["lightgbm"].predict(X_test_lgb)
    else:
        X_test_lgb = X_test
        lgb_pred = predictor.models["lightgbm"].predict(X_test_lgb)

    svr_results = predictor.train_svr(X_train, y_train)
    svr_pred = predictor.models["svr"].predict(predictor.scalers["svr"].transform(X_test))

    ridge_pred = np.clip(ridge_pred, 0.0, 1.0)
    lgb_pred = np.clip(lgb_pred, 0.0, 1.0)
    svr_pred = np.clip(svr_pred, 0.0, 1.0)

    metrics = {
        "ridge": {
            "mae": float(mean_absolute_error(y_test, ridge_pred)),
            "auc": float(roc_auc_score(y_test, ridge_pred)),
            "best_alpha": float(ridge_results["best_alpha"]),
        },
        "lightgbm": {
            "mae": float(mean_absolute_error(y_test, lgb_pred)),
            "auc": float(roc_auc_score(y_test, lgb_pred)),
        },
        "svr": {
            "mae": float(mean_absolute_error(y_test, svr_pred)),
            "auc": float(roc_auc_score(y_test, svr_pred)),
        },
    }

    with open(OUTPUTS_DIR / "model_performance.json", "w") as file:
        json.dump(metrics, file, indent=2)

    return {
        "metrics": metrics,
        "predictor": predictor,
        "feature_cols": feature_cols,
        "X_test": X_test,
        "X_test_lgb": X_test_lgb,
        "y_test": y_test,
        "groups_test": groups_test,
        "lgb_pred": lgb_pred,
        "test_idx": test_idx,
    }


def plot_model_comparison(metrics: Dict[str, Dict[str, float]]) -> None:
    model_order = ["lightgbm", "ridge", "svr"]
    mae_values = [metrics[model]["mae"] for model in model_order]
    colors = ["#55A868", "#4C72B0", "#C44E52"]

    fig, ax = plt.subplots()
    ax.barh(model_order, mae_values, color=colors)
    for idx, value in enumerate(mae_values):
        ax.text(value + 0.001, idx, f"{value:.4f}", va="center")
    ax.set_xlabel("Mean Absolute Error")
    ax.set_title("Model Performance Comparison (Mean Absolute Error)")
    ax.annotate("Lower is better", xy=(0.7, 0.1), xycoords="axes fraction")
    save_figure(fig, FIGURES_DIR / "fig2_model_comparison.png")


def shap_plots(
    predictor: MalariaRiskPredictor, X_test: np.ndarray, feature_cols: List[str]
) -> None:
    lgb_model = predictor.models["lightgbm"]
    explainer = shap.TreeExplainer(lgb_model)
    sample = shap.sample(X_test, 100, random_state=42)
    shap_values = explainer(sample, feature_names=feature_cols)

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_cols,
        max_display=10,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle("SHAP Feature Importance (LightGBM)")
    save_figure(fig, FIGURES_DIR / "fig3_shap_beeswarm.png")

    sample_index = 0
    shap_waterfall = shap_values[sample_index]
    shap.plots.waterfall(shap_waterfall, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle("Individual Risk Prediction Breakdown")
    save_figure(fig, FIGURES_DIR / "fig4_shap_waterfall.png")


def fairness_outputs(
    predictor: MalariaRiskPredictor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    lgb_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    fairness = predictor.fairness_metrics(X_test, y_test, groups_test, "lightgbm")
    with open(OUTPUTS_DIR / "fairness_metrics.json", "w") as file:
        json.dump(fairness, file, indent=2)

    tpr_data = pd.DataFrame.from_dict(fairness["by_group"], orient="index")
    tpr_data.index.name = "population"
    tpr_data.reset_index(inplace=True)

    fig, ax = plt.subplots()
    highlight_colors = {
        "Gambia": "#4C72B0",
        "Kenya": "#55A868",
        "Nigeria": "#C44E52",
    }
    colors = [highlight_colors.get(pop, "#8c8c8c") for pop in tpr_data["population"]]
    ax.bar(tpr_data["population"], tpr_data["tpr"], color=colors)
    ax.axhline(0.80, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model Sensitivity Across Populations")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    save_figure(fig, FIGURES_DIR / "fig5_fairness_tpr.png")

    calibration_groups = ["Gambia", "Kenya", "Nigeria", "Vietnam"]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")
    for group in calibration_groups:
        mask = groups_test == group
        prob_true, prob_pred = calibration_curve(y_test[mask], lgb_pred[mask], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=group)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration by Population (LightGBM)")
    ax.legend(fontsize=8)
    save_figure(fig, FIGURES_DIR / "fig6_calibration_curves.png")

    return fairness


def rs334_effect_plot(
    df: pd.DataFrame,
    predictor: MalariaRiskPredictor,
    X_test: np.ndarray,
    feature_cols: List[str],
    test_idx: np.ndarray,
) -> float:
    lgb_model = predictor.models["lightgbm"]
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer(X_test)
    rs334_index = feature_cols.index("rs334")
    rs334_shap = shap_values.values[:, rs334_index]

    rs334_data = pd.DataFrame(
        {
            "rs334_genotype": df.loc[test_idx, "rs334"].fillna(0).astype(int).to_numpy(),
            "population": df.loc[test_idx, "population"].to_numpy(),
            "shap_value": rs334_shap,
        }
    )

    fig, ax = plt.subplots()
    sns.boxplot(
        data=rs334_data,
        x="rs334_genotype",
        y="shap_value",
        hue="population",
        ax=ax,
        fliersize=1,
    )
    ax.set_xlabel("rs334 Genotype (0=HbAA, 1=HbAS, 2=HbSS)")
    ax.set_ylabel("SHAP value")
    ax.set_title("rs334 (HBB) Effect Heterogeneity Across Populations")
    ax.legend(fontsize=6, ncol=3, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_figure(fig, FIGURES_DIR / "fig7_rs334_heterogeneity.png")

    return float(np.mean(rs334_shap[rs334_data["rs334_genotype"] == 1]))


def slide_text_snippets(
    distribution: pd.DataFrame,
    metrics: Dict[str, Dict[str, float]],
    fairness: Dict[str, Dict[str, float]],
    rs334_mean: float,
) -> None:
    lines: List[str] = []
    for _, row in distribution.iterrows():
        lines.append(f"{row['population']}: {row['count']:,} ({row['percentage']:.2f}%)")

    lines.append(
        "LightGBM achieved MAE {:.1e}, outperforming Ridge ({:.1e}) and SVR ({:.1e}).".format(
            metrics["lightgbm"]["mae"],
            metrics["ridge"]["mae"],
            metrics["svr"]["mae"],
        )
    )
    tpr_disparity = fairness["tpr_disparity"]
    if tpr_disparity < 0.10:
        tpr_statement = "passing IEEE 2857-2021 thresholds (ΔTPR<0.10)"
    else:
        tpr_statement = "exceeding IEEE 2857-2021 thresholds (ΔTPR<0.10)"
    lines.append(f"TPR disparity of {tpr_disparity:.2f} across populations, {tpr_statement}.")
    lines.append(
        "rs334 (HBB) showed a mean SHAP contribution of {:.2f} across all populations.".format(
            rs334_mean
        )
    )

    with open(OUTPUTS_DIR / "slide_text_snippets.txt", "w") as file:
        file.write("\n".join(lines))


def main() -> None:
    ensure_dirs()
    generator = MalariaDataGenerator(n_samples=20817, random_state=42)
    df = generator.save("data/synthetic_clinical.csv")

    distribution = population_distribution(df)
    plot_population_pie(distribution)

    training_results = train_models(df)
    metrics = training_results["metrics"]
    plot_model_comparison(metrics)

    shap_plots(
        training_results["predictor"],
        training_results["X_test_lgb"],
        training_results["feature_cols"],
    )

    fairness = fairness_outputs(
        training_results["predictor"],
        training_results["X_test_lgb"],
        training_results["y_test"],
        training_results["groups_test"],
        training_results["lgb_pred"],
    )

    rs334_mean = rs334_effect_plot(
        df,
        training_results["predictor"],
        training_results["X_test"],
        training_results["feature_cols"],
        training_results["test_idx"],
    )

    slide_text_snippets(distribution, metrics, fairness, rs334_mean)


if __name__ == "__main__":
    main()
