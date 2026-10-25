"""CLI entry point for IDFS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Demand Forecasting System",
        prog="idfs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate synthetic data
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated data",
    )

    args = parser.parse_args()

    if args.command == "generate":
        _cmd_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic data command."""
    from idfs.config import load_config
    from idfs.data.synthetic import (
        SyntheticConfig,
        generate_synthetic_data,
        validate_synthetic_schema,
    )

    cfg = load_config(args.config)
    syn_cfg = SyntheticConfig(
        n_stores=cfg.data.synthetic_n_series // 10 or 5,
        n_families=min(cfg.data.synthetic_n_series, 10),
        n_days=cfg.data.synthetic_n_days,
        seed=cfg.data.synthetic_seed,
        intermittent_frac=cfg.data.synthetic_intermittent_frac,
        drift_start_day=cfg.data.synthetic_drift_start_day,
    )

    print(
        f"Generating synthetic data: {syn_cfg.n_stores} stores x "
        f"{syn_cfg.n_families} families x {syn_cfg.n_days} days"
    )
    data = generate_synthetic_data(syn_cfg)

    errors = validate_synthetic_schema(data)
    if errors:
        print(f"Schema validation errors: {errors}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  Wrote {name}: {len(df)} rows -> {path}")

    print("Done!")


if __name__ == "__main__":
    main()
