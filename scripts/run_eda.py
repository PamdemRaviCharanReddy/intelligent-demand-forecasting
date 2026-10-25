"""Standalone script to execute data pipeline and generate EDA figures."""

from __future__ import annotations

import argparse

from idfs.data.eda import generate_eda_figures
from idfs.data.fusion import run_data_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDFS Data Ingestion and EDA")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/synthetic",
        help="Path to raw source directory (defaults to data/synthetic)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/fused_sales.parquet",
        help="Path for processed fused Parquet dataset",
    )
    args = parser.parse_args()

    print(f"Running data pipeline from: {args.source_dir}")
    fused_df = run_data_pipeline(args.source_dir, args.output_path)
    print(f"Fused dataset shape: {fused_df.shape} -> saved to {args.output_path}")

    print("Generating EDA figures...")
    figures = generate_eda_figures(fused_df)
    for name, path in figures.items():
        print(f"  Generated {name}: {path}")

    print("EDA complete!")


if __name__ == "__main__":
    main()
