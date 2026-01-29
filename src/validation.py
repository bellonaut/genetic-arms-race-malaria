import json
from pathlib import Path

import great_expectations as ge


def validate_dataset(df, output_path: Path | str | None = None):
    """Run lightweight GE checks tailored to the synthetic dataset."""

    ge_df = ge.from_pandas(df)

    # Row count must match target total
    ge_df.expect_table_row_count_to_equal(20817)

    # Genotype bounds
    ge_df.expect_column_values_to_be_between("rs334", 0, 2)

    # Risk score magnitude
    ge_df.expect_column_values_to_be_between("wGRS_GF_POS", 1e-7, 1e-3, mostly=0.99)

    result = ge_df.validate()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    return result
