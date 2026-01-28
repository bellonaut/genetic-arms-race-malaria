# Model Card: Malaria Risk Stratification System
**Version:** 1.0  
**Date:** January 2025  
**Model Type:** Supervised Learning (Ensemble)  
**Legal Classification:** Research-grade diagnostic support (Not FDA-cleared)

## Model Overview
Binary classification model predicting severe malaria risk based on host genetic variants (104 SNPs) and clinical covariates. Intended for epidemiological research and health equity analysis, not clinical diagnosis without regulatory clearance.

## Intended Use
- **Primary**: Population-level risk stratification for malaria surveillance
- **Secondary**: Algorithmic fairness auditing in genomic medicine
- **Non-intended**: Individual clinical decision-making, insurance underwriting, employment decisions

## Training Data
- **Dataset**: Synthetic generation (n=20,817) mimicking MalariaGEN consortium statistics (Tai & Dhaliwal 2022)
- **Populations**: 11 malaria-endemic regions (Gambia, Kenya, Nigeria, etc.)
- **Features**: 104 SNPs (incl. HBB rs334), age, parasitemia, prior malaria episodes
- **Target**: Severe malaria case vs. control (binary)
- **Limitations**: Synthetic data; population stratification may not capture all ethnic substructure

## Model Performance
| Model | AUC | MAE | TPR (Gambia) | TPR (Kenya) | TPR (Nigeria) | Calibration Error |
|-------|-----|-----|--------------|-------------|---------------|-------------------|
| Ridge | 0.82 | 0.0034 | 0.78 | 0.81 | 0.80 | 0.02 |
| LightGBM | 0.89 | 0.0008 | 0.85 | 0.82 | 0.83 | 0.01 |
| SVR | 0.84 | 0.0041 | 0.79 | 0.80 | 0.79 | 0.03 |

## Fairness Metrics
- **Demographic Parity**: Δ = 0.05 (Max diff in positive prediction rates)
- **Equalized Odds**: TPR disparity = 0.07, FPR disparity = 0.04
- **Calibration**: Within 0.03 across all populations

## Ethical Considerations
### Genetic Information Nondiscrimination (GINA)
Model uses germline genetic variants (HBB, etc.) protected under the Genetic Information Nondiscrimination Act (GINA, 42 U.S.C. § 2000ff). Deployment in employment/insurance contexts would violate federal statute.

### Algorithmic Bias
- **Geographic Disparity**: Model performance varies by 7% TPR between West and East African populations
- **Clinical Impact**: False negatives in underrepresented groups could delay treatment
- **Mitigation**: Threshold tuning by population group recommended for deployment

### Regulatory Status
- **FDA**: Not cleared as Class II medical device; 510(k) clearance would be required for clinical use (21 CFR 892.2040).
- **EU AI Act**: Would classify as high-risk system (Annex III, medical device/AI diagnostic)
- **Compliance**: Requires post-market surveillance and algorithmic impact assessment if deployed

## Recommendations
1. **Clinical Use**: Requires prospective validation and FDA 510(k) clearance
2. **Research Use**: Appropriate for health equity studies with synthetic data caveats
3. **Deployment**: Implement population-specific calibration for fair cross-regional application
