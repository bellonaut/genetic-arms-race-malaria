"""Evolutionary modeling utilities for sickle-cell malaria co-evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
