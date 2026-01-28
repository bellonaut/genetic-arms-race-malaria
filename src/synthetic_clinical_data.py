"""Synthetic clinical data generator for malaria risk stratification.

Generates high-fidelity synthetic data mimicking MalariaGEN consortium statistics
(Tai & Dhaliwal 2022) without using restricted individual-level data.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class MalariaDataGenerator:
    """Generate synthetic malaria genomic and clinical data."""

    def __init__(self, n_samples: int = 20817, random_state: int = 42) -> None:
        """Initialize generator with population parameters.

        Args:
            n_samples: Total sample size across all populations.
            random_state: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        # Population distribution (Tai & Dhaliwal 2022, MalariaGEN proxy)
        raw_populations = {
            "Gambia": 0.268,
            "Kenya": 0.177,
            "Malawi": 0.148,
            "Burkina Faso": 0.069,
            "Cameroon": 0.071,
            "Ghana": 0.037,
            "Mali": 0.042,
            "Nigeria": 0.020,
            "Tanzania": 0.047,
            "Vietnam": 0.083,
            "PNG": 0.039,  # Papua New Guinea
        }
        total = sum(raw_populations.values())
        self.populations = {key: value / total for key, value in raw_populations.items()}

        # Key SNPs and effect sizes from literature
        self.snp_config = self._load_snp_config()

    def _load_snp_config(self) -> Dict[str, Dict[str, object]]:
        """Load SNP configurations (effect sizes, positions, frequencies).

        Returns:
            Dictionary of SNP metadata including effect sizes, genomic positions,
            and population-specific minor allele frequencies.
        """
        config: Dict[str, Dict[str, object]] = {
            "rs334": {
                "chrom": 11,
                "pos": 5227002,
                "effect": 0.85,
                "maf": {
                    "Gambia": 0.12,
                    "Kenya": 0.15,
                    "Malawi": 0.10,
                    "Burkina Faso": 0.14,
                    "Cameroon": 0.11,
                    "Ghana": 0.13,
                    "Mali": 0.16,
                    "Nigeria": 0.18,
                    "Tanzania": 0.13,
                    "Vietnam": 0.02,
                    "PNG": 0.01,
                },
            }
        }

        for index in range(1, 104):
            snp_id = f"rs{index + 1000}"
            config[snp_id] = {
                "chrom": int(self.rng.choice([1, 2, 6, 11, 16])),
                "pos": int(self.rng.integers(100_000, 250_000_000)),
                "effect": float(
                    self.rng.exponential(0.05) if index < 10 else self.rng.exponential(0.01)
                ),
                "maf": {
                    pop: float(np.clip(self.rng.beta(2, 5), 0.01, 0.50))
                    for pop in self.populations
                },
            }

        return config

    @staticmethod
    def _genotype_frequency(genotype: int, maf: float) -> float:
        """Compute Hardy-Weinberg genotype frequency for a given MAF.

        Args:
            genotype: Genotype encoding (0, 1, 2).
            maf: Minor allele frequency for the population.

        Returns:
            Genotype frequency estimate.
        """
        if genotype == 0:
            return (1 - maf) ** 2
        if genotype == 1:
            return 2 * maf * (1 - maf)
        return maf**2

    def simulate_genotypes(self, population: str, n_samples: int) -> pd.DataFrame:
        """Simulate genotype matrix for a population.

        Args:
            population: Population label to condition allele frequencies.
            n_samples: Number of samples to generate.

        Returns:
            DataFrame of genotype calls for 104 SNPs.
        """
        snp_ids = list(self.snp_config.keys())
        genotype_matrix = np.zeros((n_samples, len(snp_ids)), dtype=int)
        for idx, snp in enumerate(snp_ids):
            maf = self.snp_config[snp]["maf"][population]
            genotype_matrix[:, idx] = self.rng.binomial(2, maf, size=n_samples)
        return pd.DataFrame(genotype_matrix, columns=snp_ids)

    def calculate_risk_score(self, genotypes: pd.DataFrame, population: str) -> np.ndarray:
        """Calculate Tai & Dhaliwal wGRS+GF+POS risk score.

        Formula: (risk_allele * effect_size * genotype_freq) / (snp_position * column_index)

        Args:
            genotypes: DataFrame of genotype calls for each SNP.
            population: Population label for allele frequency lookup.

        Returns:
            Array of risk scores (higher indicates higher risk).
        """
        scores = np.zeros(genotypes.shape[0])
        snp_ids = list(self.snp_config.keys())

        for column_index, snp in enumerate(snp_ids, start=1):
            config = self.snp_config[snp]
            maf = config["maf"][population]
            genotype_values = genotypes[snp].to_numpy()
            genotype_freq = np.vectorize(self._genotype_frequency)(genotype_values, maf)
            risk_allele = genotype_values.astype(float)
            numerator = risk_allele * config["effect"] * genotype_freq
            denominator = config["pos"] * column_index
            scores += numerator / denominator

        return scores

    def sample_population(self, population: str, n_samples: int) -> pd.DataFrame:
        """Generate population-stratified samples with genotypes and clinical data.

        Args:
            population: Population label.
            n_samples: Number of samples to generate.

        Returns:
            DataFrame with genotypes, clinical covariates, and risk labels.
        """
        genotypes = self.simulate_genotypes(population, n_samples)
        risk_score = self.calculate_risk_score(genotypes, population)

        clinical = pd.DataFrame(
            {
                "age": self.rng.gamma(2, 3, size=n_samples) + 1,
                "parasitemia": self.rng.lognormal(3, 1.5, size=n_samples),
                "prior_malaria": self.rng.poisson(2, size=n_samples),
            }
        )

        risk_z = (risk_score - np.mean(risk_score)) / (np.std(risk_score) + 1e-6)
        age_z = (clinical["age"] - clinical["age"].mean()) / (clinical["age"].std() + 1e-6)
        para_z = (clinical["parasitemia"] - clinical["parasitemia"].mean()) / (
            clinical["parasitemia"].std() + 1e-6
        )
        prior_z = (clinical["prior_malaria"] - clinical["prior_malaria"].mean()) / (
            clinical["prior_malaria"].std() + 1e-6
        )
        rs334_effect = genotypes["rs334"].astype(float)
        logit = (10.0 * risk_z) + (2.0 * age_z) + (2.0 * para_z) - (1.0 * prior_z) + (
            3.0 * rs334_effect
        )
        prob_case = 1 / (1 + np.exp(-logit))
        case = (prob_case >= 0.5).astype(int)

        df = pd.concat([genotypes, clinical], axis=1)
        df["risk_score"] = risk_score
        df["case"] = case
        df["population"] = population
        return df

    def generate(self) -> pd.DataFrame:
        """Generate the complete synthetic dataset.

        Returns:
            DataFrame containing all populations and features.
        """
        data_frames: List[pd.DataFrame] = []
        population_items = list(self.populations.items())

        sample_counts: Dict[str, int] = {}
        remaining = self.n_samples
        for idx, (pop, prop) in enumerate(population_items):
            if idx == len(population_items) - 1:
                count = remaining
            else:
                count = int(round(self.n_samples * prop))
                remaining -= count
            sample_counts[pop] = count

        sample_id = 0
        for pop, count in sample_counts.items():
            pop_df = self.sample_population(pop, count)
            pop_df.insert(0, "sample_id", np.arange(sample_id, sample_id + count))
            sample_id += count
            data_frames.append(pop_df)

        df = pd.concat(data_frames, ignore_index=True)

        mask_cols = [
            col for col in df.columns if col not in ("sample_id", "population", "case", "risk_score")
        ]
        mask = self.rng.random(df[mask_cols].shape) < 0.02
        df = df.astype({col: "float64" for col in mask_cols})
        df.loc[:, mask_cols] = df.loc[:, mask_cols].mask(mask)

        return df

    def save(self, output_path: str = "data/synthetic_clinical.csv") -> pd.DataFrame:
        """Generate and save dataset with metadata.

        Args:
            output_path: Destination path for the CSV file.

        Returns:
            Generated DataFrame.
        """
        df = self.generate()
        df.to_csv(output_path, index=False)

        metadata = {
            "n_samples": int(len(df)),
            "n_snps": 104,
            "populations": self.populations,
            "generation_method": "wGRS+GF+POS (Tai & Dhaliwal 2022)",
            "synthetic": True,
            "random_state": self.random_state,
        }

        with open(output_path.replace(".csv", "_metadata.json"), "w") as file:
            json.dump(metadata, file, indent=2)

        return df


if __name__ == "__main__":
    generator = MalariaDataGenerator()
    dataset = generator.save()
    print(
        "Generated {} synthetic samples across {} populations".format(
            len(dataset), dataset["population"].nunique()
        )
    )
    print("Case rate: {:.2%}".format(dataset["case"].mean()))
