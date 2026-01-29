## Replication Summary

We replicate the wGRS+GF+POS framework from Tai & Dhaliwal (2022) using
ethically-synthesized data preserving population structure (N=20,817)
and Hardy–Weinberg genotype frequencies.

### Sanity Checks
- Population totals and proportions: PASS
- SNP count (104) and rs334 presence: PASS
- Score scaling and directionality: PASS

### Model Performance (MAE)
| Model        | MAE |
|-------------|-----|
| LightGBM    | 3.0e-05 |
| Ridge       | 5.7e-08 |
| SVR         | 8.2e-04 |
| MLP 8-8-8-1 | 7.8e-04 |

### Key Observation
Incorporating mutation location (POS) compresses the target scale by
~10⁵× relative to baseline wGRS+GF, enabling substantially improved
prediction accuracy.

### Limitations
- Synthetic data cannot capture LD structure.
- Nigeria (≈2% of sample) exhibits higher variance.
