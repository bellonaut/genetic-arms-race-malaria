#!/usr/bin/env python3
"""Generate IEEE 2857-2021 fairness audit metrics for presentation slides."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def main() -> None:
    output_dir = Path("outputs")
    df = pd.read_csv(output_dir / "synthetic_clinical.csv")
    _ = json.load(open(output_dir / "model_metrics.json"))

    feature_cols = [c for c in df.columns if c.startswith("rs") and not c.startswith("maf_")]
    X = df[feature_cols].to_numpy()
    y = df["wGRS_GF_POS"].to_numpy()
    groups = df["Population"].to_numpy()

    median_risk = np.median(y)
    y_true_binary = (y > median_risk).astype(int)

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X,
        y_true_binary,
        groups,
        test_size=0.15,
        random_state=42,
        stratify=groups,
    )

    model = LGBMRegressor(
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
    )
    model.fit(X_train, y_train)
    y_pred_continuous = model.predict(X_test)
    y_pred_binary = (y_pred_continuous > np.median(y_pred_continuous)).astype(int)

    results = {}
    populations = np.unique(groups_test)

    for pop in populations:
        mask = groups_test == pop
        y_true_pop = y_test[mask]
        y_pred_pop = y_pred_binary[mask]

        tn, fp, fn, tp = confusion_matrix(y_true_pop, y_pred_pop, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        demographic_parity = float(np.mean(y_pred_pop))

        results[pop] = {
            "n_samples": int(np.sum(mask)),
            "TPR": float(tpr),
            "FPR": float(fpr),
            "TNR": float(tnr),
            "PPV": float(ppv),
            "Demographic_Parity": float(demographic_parity),
            "Prevalence": float(np.mean(y_true_pop)),
        }

    tpr_values = [results[p]["TPR"] for p in populations]
    dp_values = [results[p]["Demographic_Parity"] for p in populations]

    dp_min = min(dp_values)
    dp_max = max(dp_values)
    demographic_parity_ratio = dp_min / dp_max if dp_max > 0 else 0

    tpr_max_diff = max(tpr_values) - min(tpr_values)

    fpr_values = [results[p]["FPR"] for p in populations]
    fpr_max_diff = max(fpr_values) - min(fpr_values)
    equalized_odds_delta = max(tpr_max_diff, fpr_max_diff)

    calibration_errors = []
    for pop in populations:
        mask = groups_test == pop
        predicted_risk = np.mean(y_pred_binary[mask])
        actual_risk = np.mean(y_test[mask])
        calibration_errors.append(abs(predicted_risk - actual_risk))

    mean_calibration_error = np.mean(calibration_errors)

    slide_data = {
        "Demographic_Parity_Ratio": round(demographic_parity_ratio, 2),
        "Passes_4_Fifths_Rule": "PASS" if demographic_parity_ratio > 0.8 else "FAIL",
        "TPR_Disparity_Max": round(tpr_max_diff, 2),
        "TPR_Range": f"{min(tpr_values):.2f} vs {max(tpr_values):.2f}",
        "Equalized_Odds_Delta": round(equalized_odds_delta, 2),
        "Mean_Calibration_Error": round(mean_calibration_error, 3),
        "Populations": {
            k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
            for k, v in results.items()
        },
    }

    print("=" * 60)
    print("ALGORITHMIC FAIRNESS AUDIT (IEEE 2857-2021)")
    print("=" * 60)
    print(f"\nDemographic Parity Ratio: {slide_data['Demographic_Parity_Ratio']}")
    print(f"Passes 4/5ths Rule (>0.80): {slide_data['Passes_4_Fifths_Rule']}")
    print(f"\nTPR Disparity (Max): {slide_data['TPR_Disparity_Max']}")
    print(f"TPR Range: {slide_data['TPR_Range']}")
    print(f"\nEqualized Odds (ΔFPR/ΔTPR): {slide_data['Equalized_Odds_Delta']}")
    print(f"Mean Calibration Error: {slide_data['Mean_Calibration_Error']}")

    print("\nPer-Population TPR (Sensitivity):")
    for pop in sorted(populations, key=lambda x: results[x]["TPR"], reverse=True):
        tpr = results[pop]["TPR"]
        n = results[pop]["n_samples"]
        print(f"  {pop:15s}: {tpr:.2f} (n={n})")

    with open(output_dir / "fairness_audit_slide.json", "w") as f:
        json.dump(slide_data, f, indent=2)

    print(f"\nSaved to {output_dir / 'fairness_audit_slide.json'}")

    fig, ax = plt.subplots(figsize=(10, 6))

    pop_names = sorted(populations, key=lambda x: results[x]["TPR"])
    tpr_vals = [results[p]["TPR"] for p in pop_names]
    colors = [
        "#2E2E2E" if tpr > 0.75 else "#666666" if tpr > 0.7 else "#999999"
        for tpr in tpr_vals
    ]

    bars = ax.barh(pop_names, tpr_vals, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Sensitivity Across Geographic Populations", fontsize=14, fontweight="bold", pad=20)

    for bar, val in zip(bars, tpr_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=10)

    ax.axvline(x=0.80, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.80, len(pop_names) - 0.5, "80% threshold", color="red", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "fairness_chart.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fairness_chart.svg", format="svg")
    print(f"Saved chart to {output_dir / 'fairness_chart.png'}")

    print("\n" + "=" * 60)
    print("WARNINGS FOR SLIDE:")
    if any(results[p]["TPR"] < 0.75 for p in populations):
        low_tpr = [p for p in populations if results[p]["TPR"] < 0.75]
        print(f"  ⚠ LOW TPR detected in: {', '.join(low_tpr)}")
        print("    Explain: Small sample size or population stratification")

    if demographic_parity_ratio < 0.8:
        print(f"  ⚠ FAILS 4/5ths Rule: {demographic_parity_ratio:.2f} < 0.80")
    else:
        print(f"  ✓ Passes 4/5ths Rule: {demographic_parity_ratio:.2f} >= 0.80")


if __name__ == "__main__":
    main()
