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

## Getting Started
```bash
pip install -r requirements.txt
jupyter notebook notebooks/parasite_genotype_analysis.ipynb
```

## License
MIT License. See `LICENSE`.

[^1^]: Band, G., Leffler, E. M., Jallow, M., et al. *Nature* (2022). See `references.bib`.
[^2^]: Kwiatkowski, D. P. *American Journal of Human Genetics* (2005). See `references.bib`.
[^3^]: World Health Organization. *World Malaria Report 2022* (2022). See `references.bib`.


![Genetic Arms Race](genetic-arms-race-title.png)

# Sickle-Cell Malaria Co-evolution
...
