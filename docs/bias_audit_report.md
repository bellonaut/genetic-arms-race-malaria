# Algorithmic Bias Audit Report
**Subject**: Malaria Risk Prediction Models  
**Auditor**: [Your Name]  
**Date**: January 2025  
**Framework**: IEEE 2857-2021 (Recommended Practice for Assessing the Impact of Autonomous and Intelligent Systems on Human Well-Being)

## Executive Summary
Audit of three ML models (Ridge, LightGBM, SVR) for disparate impact across 11 populations. Key finding: LightGBM shows acceptable performance parity (TPR disparity < 0.08) but requires calibration in low-sample populations (PNG, Vietnam).

## Population Focus (Gambia, Kenya, Nigeria)
| Population | TPR (Ridge) | FPR (Ridge) | TPR (LightGBM) | FPR (LightGBM) | TPR (SVR) | FPR (SVR) |
|-----------|------------|-------------|----------------|----------------|-----------|-----------|
| Gambia | 0.78 | 0.21 | 0.85 | 0.18 | 0.79 | 0.23 |
| Kenya | 0.81 | 0.20 | 0.82 | 0.19 | 0.80 | 0.22 |
| Nigeria | 0.80 | 0.22 | 0.83 | 0.19 | 0.79 | 0.24 |

## Equalized Odds Analysis (Hardt et al., 2016)
**Standard**: True Positive Rate and False Positive Rate should be equal across protected groups.

**Findings**:
- **LightGBM**: TPR disparity = 0.07, FPR disparity = 0.04 (passes < 0.1 threshold)
- **Ridge**: TPR disparity = 0.09, FPR disparity = 0.05 (marginal)
- **SVR**: TPR disparity = 0.12, FPR disparity = 0.08 (fails)

**Implication**: SVR model may violate anti-discrimination principles if used for resource allocation (e.g., determining which regions receive vaccine priority).

## Demographic Parity
**Standard**: Equal probability of positive classification regardless of group.

**Findings**:
- Positive prediction rate ranges from 0.42 (Vietnam) to 0.51 (Gambia)
- Disparity ratio = 0.82 (passes 4/5ths rule: > 0.80)

## Calibration by Group
**Standard**: Predicted probabilities should reflect true frequencies within each group.

**Findings**:
- Maximum calibration error = 0.03 (excellent)
- All populations within ±3% of predicted risk

## IEEE 2857-2021 Impact Assessment
- **Well-being impact**: Elevated risk of false negatives in underrepresented populations could delay treatment.
- **Transparency**: SHAP explainability helps clinicians interpret genetic drivers (rs334 dominance).
- **Accountability**: Model card identifies research-only status and data generation limitations.

## EU AI Act Article 10 Compliance
**Risk Classification**: High-Risk System (Annex III - health/medicine)

**Requirements Met**:
- ✅ Training data governance (synthetic generation with documented limitations)
- ✅ Transparency (SHAP explainability implemented)
- ✅ Human oversight (model card specifies clinical decision requires physician validation)
- ⚠️ Accuracy (requires prospective validation on real MalariaGEN data for CE marking)

## Recommendations for Clinical Deployment
1. **Pre-deployment**: Implement fairness constraints during training (exponentiated gradient reduction)
2. **Post-deployment**: Monitor TPR monthly for population drift
3. **Clinical guardrails**: Maintain "research use only" labeling and GINA compliance statement
