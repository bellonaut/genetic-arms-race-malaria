![Genetic Arms Race](<img width="4000" height="2250" alt="genetic arms race_sickle_cell_vs_malaria - Copy-3" src="https://github.com/user-attachments/assets/f9727fb8-57da-4806-9edf-9ae5b10f074b" />
)
# Sickle-Cell Malaria Co-evolution

## Abstract
This repository provides a hybrid literature review and computational synthesis of parasite genotype-specific malaria resistance in HbAS (sickle cell trait) carriers, centered on Band et al. (2022).[^1^] We integrate evolutionary theory, mechanistic biology, and synthetic analyses to illustrate how *Plasmodium falciparum* genotypes modulate protection, and why these dynamics matter for global health in Sub-Saharan Africa. The project is designed for graduate-level researchers and interdisciplinary audiences in human genetics, infectious disease, and health policy.

## Key Findings Synthesized
- Parasite genotype-specific protection: HbAS protection is attenuated for Pfsa+ infections (RR ≈ 0.83, CI 0.53–1.30) and profoundly protective for Pfsa− infections (RR ≈ 0.01, CI 0.007–0.03).[^1^]
- Pfsa1, Pfsa2, and Pfsa3 loci show strong associations with HbAS-specific effects, with candidate genes including PfACS8 and export protein regions.[^1^]
- The findings support a host–parasite evolutionary arms race, consistent with broader frameworks in malaria population genetics.[^2^]

## Global Health Context
The WHO reported approximately 249 million malaria cases and 608,000 deaths in 2022, with the heaviest burden in Sub-Saharan Africa.[^3^] These statistics underscore the urgency of understanding parasite adaptation to host protective polymorphisms.

## Repository Structure
- `notebooks/`: Jupyter notebook with comparative risk ratio visualizations, forest plots, linkage disequilibrium schematics, and Plotly interactivity.
- `docs/`: Academic literature synthesis and mechanistic narratives with citations.
- `data/`: Synthetic data files used for visualization and modeling.
- `src/`: Computational utilities for heterozygote advantage modeling and mock GWAS visualization.

## Reproducibility
The computational synthesis relies on reported effect sizes and confidence intervals from primary literature (especially Band et al. 2022) and converts them into synthetic datasets for visualization and exploratory modeling.[^1^] The notebook intentionally uses simulated data to avoid proprietary or sensitive datasets while preserving the reported directional effects and confidence intervals.

## Machine Learning Expansion (v2.0)

This repository now includes algorithmic risk stratification and fairness auditing capabilities:

### New Components
- **Synthetic Clinical Data Generator**: High-fidelity simulation of 20,817 individuals across 11 populations using Tai & Dhaliwal (2022) wGRS+GF+POS methodology
- **Multi-Model Risk Prediction**: Ridge Regression, LightGBM, and SVR with Genetic Algorithm-simulated hyperparameter tuning
- **Algorithmic Fairness Audit**: Disparate impact analysis across Gambia, Kenya, Nigeria, and other populations per IEEE 2857-2021 standards
- **SHAP Explainability**: Model interpretation showing rs334 (HBB) dominance and mutation location effects
- **Regulatory Documentation**: FDA-style model cards and EU AI Act compliance assessment

### Legal & Policy Frameworks
The ML module specifically addresses:
- **GINA Compliance**: Genetic information nondiscrimination safeguards
- **EU AI Act**: High-risk system classification and requirements for medical AI
- **Algorithmic Fairness**: Equalized odds and demographic parity across African populations
- **Health Equity**: Calibration assessment ensuring model validity in low-resource Nigerian contexts (NaijaCare integration)

## Results at a Glance
- **Sample size / populations**: 20,817 synthetic individuals across 11 regions (Nigeria = 423, 2.03%).
- **Model MAE (wGRS+GF+POS)**: LightGBM 3.0e-05; Ridge 5.7e-08; SVR 8.2e-04; MLP 7.8e-04. (Source: `outputs/model_metrics.json`.)
- **Fairness (LightGBM, per IEEE 2857-2021)**: Demographic Parity Ratio 1.00; TPR disparity 0.06 (0.94–0.99); Equalized Odds Δ 0.06; calibration error 0.0. (Source: `outputs/fairness_audit_slide.json` / `outputs/fairness_chart.png`.)
- **Effect direction check**: rs334 shows negative correlation with risk score (≈ -0.58), matching protective expectation.


## What I Did
I extended my earlier biological reading of Band et al. (2022) into a working machine-learning system by implementing the position-weighted genetic risk framework of Tai & Dhaliwal (2022). Starting from population constraints rather than clean labels, I built an end-to-end pipeline on ethically generated, population-structured synthetic data (n = 20,817): simulating genotypes under Hardy-Weinberg equilibrium, engineering risk features, training multiple models, and stress-testing numerical behavior across populations. Much of the work involved debugging scale, leakage-like artifacts, and pipeline assumptions until the outputs behaved in ways that were biologically and statistically coherent.

## What I Learned
This project taught me that most of the difficulty in applied ML lives before model training. Synthetic data is deceptively fragile: small choices in scaling, denominators, or feature construction can produce metrics that look impressive but mean nothing. I learned to treat near-perfect scores as warnings rather than victories, to use simple models as diagnostic tools rather than endpoints, and to rely on sanity checks, population-level error analysis, and explicit failure modes to understand what a system is actually learning. More than anything, it clarified the difference between reproducing a signal and building something that could ever survive contact with real data.

## Data Dictionary / Schema (synthetic_clinical.csv)
- `Population` (str): one of 11 malaria-endemic regions.
- `rs334`, `rs_1` … `rs_103` (int): genotype calls encoded 0/1/2.
- `maf_rs334`, `maf_rs_1` … `maf_rs_103` (float): population-specific MAF used to compute genotype frequencies.
- `wGRS_GF_POS` (float): Tai & Dhaliwal weighted genetic risk score including position term (target for models).
- `wGRS_GF` (float): baseline weighted genetic risk score without position term.
- (Additional engineered features appear only in notebooks, not in the saved CSV.)

## Ethical & Governance Framing
- **Protected variant handling**: rs334 (sickle cell) is protected under GINA; results are synthetic and for research/demo only.
- **Regulatory posture**: Framed as research-use software; EU AI Act high-risk classification noted; no clinical deployment implied.
- **Fairness choices**: Per-population thresholds used to equalize positive rates; fairness metrics reported openly with calibration error.

### Running the Analysis
```bash
pip install -r requirements.txt
python src/synthetic_clinical_data.py
jupyter notebook notebooks/02_ml_risk_stratification.ipynb
jupyter notebook notebooks/03_fairness_audit.ipynb
```

## Getting Started
```bash
pip install -r requirements.txt
python scripts/plot_pos_gain.py
jupyter notebook notebooks/parasite_genotype_analysis.ipynb
```

## License
MIT License. See `LICENSE`.

[^1^]: Band, G., Leffler, E. M., Jallow, M., et al. *Nature* (2022). See `references.bib`.
[^2^]: Kwiatkowski, D. P. *American Journal of Human Genetics* (2005). See `references.bib`.
[^3^]: World Health Organization. *World Malaria Report 2022* (2022). See `references.bib`.



# Sickle-Cell Malaria Co-evolution
...
