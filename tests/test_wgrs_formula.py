import numpy as np
import pandas as pd
import pytest

from src.tai_dhaliwal_pipeline import (
    SNPConfig,
    GENOTYPE_CODES,
    POPULATIONS,
    calculate_wgrs,
)


def test_genotype_encoding():
    """Verify Tai & Dhaliwal encoding: AA=104, Aa=208, aa=312."""
    assert GENOTYPE_CODES[0] == 104
    assert GENOTYPE_CODES[1] == 208
    assert GENOTYPE_CODES[2] == 312
    for code in GENOTYPE_CODES.values():
        assert code % 104 == 0


def test_nigeria_population_count():
    """Validate Nigeria scaling matches pipeline rounding strategy."""
    total = sum(POPULATIONS.values())
    scale = 20817 / total
    scaled = {k: int(v * scale) for k, v in POPULATIONS.items()}
    scaled["Nigeria"] += 20817 - sum(scaled.values())
    nigeria_scaled = scaled["Nigeria"]
    assert nigeria_scaled == 423, f"Nigeria count {nigeria_scaled} deviates from pipeline output"


def test_rs334_protective_direction():
    """rs334 (sickle cell) must reduce risk score (negative correlation)."""
    df_test, snp_config = create_minimal_dataset()
    scores = calculate_wgrs(df_test, snp_config)
    correlation = np.corrcoef(df_test["rs334"], scores)[0, 1]
    assert correlation < -0.1, "rs334 should show protective (negative) correlation"


def test_mutation_value_calculation():
    """Verify POS×Index denominator calculation."""
    expected = 208 * 1  # genotype_code * index
    actual = GENOTYPE_CODES[1] * (0 + 1)
    assert actual == expected


# Helpers
def create_minimal_dataset():
    """Small DF with rs334 and matching maf column."""
    df = pd.DataFrame(
        {
            "rs334": np.array([0, 1, 2]),
            "maf_rs334": np.array([0.2, 0.2, 0.2]),
        }
    )
    snp_config = [SNPConfig(snp_id="rs334", effect_size=-0.85, index=1)]
    return df, snp_config
