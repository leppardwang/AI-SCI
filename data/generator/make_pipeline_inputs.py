#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write low-memory column projections of the rebuilt corpus.

The pipeline scripts (`precompute_metrics.py`, `train_lstm.py`) call
`pd.read_csv(path)` with no column selection, so their peak RSS is driven by
the number of columns. On this host the free commit charge is < 1 GB, so the
audited runs are fed exact column projections of the SAME rows/values:

  * ..._pipeline10.csv -> the 10 columns backend/engine.py requires
    (used for precompute_metrics.py, so the JSON is bit-identical to a run on
    the 23-column file)
  * ..._lstm6.csv       -> the 6 columns backend/features.py + train_lstm.py read

Row values are copied verbatim; only columns are dropped. Streamed in chunks,
so peak memory stays ~60 MB.

    python make_pipeline_inputs.py [--csv SRC] [--chunk 50000] [--tag TAG]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

REQUIRED = ["timestamp", "hospital_id", "volume_kb", "latency_ms", "risk_label",
            "risk_type", "action", "response_status", "is_trusted_line",
            "dest_platform"]
LSTM_COLS = ["timestamp", "volume_kb", "latency_ms", "risk_label",
             "is_trusted_line", "dest_platform"]


def project(src: Path, cols: list[str], out: Path, chunk: int) -> None:
    first = True
    n = 0
    for part in pd.read_csv(src, usecols=cols, chunksize=chunk,
                            low_memory=False):
        part.to_csv(out, index=False, mode="w" if first else "a",
                    header=first, encoding="utf-8", lineterminator="\n")
        first = False
        n += len(part)
    print(f"  {out.name}: {n:,} rows x {len(cols)} cols, "
          f"{out.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "lecheng_security_logs_2024H1.csv"))
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--tag", default="",
                    help="inserted into the projection file names, e.g. "
                         "'variantB' -> lecheng_security_logs_2024H1_variantB_"
                         "pipeline10.csv. Empty keeps the variant A names.")
    args = ap.parse_args()
    src = Path(args.csv)
    stem = f"lecheng_security_logs_2024H1{('_' + args.tag) if args.tag else ''}"
    print(f"projecting {src.name} -> stem {stem}")
    project(src, REQUIRED, src.with_name(f"{stem}_pipeline10.csv"), args.chunk)
    project(src, LSTM_COLS, src.with_name(f"{stem}_lstm6.csv"), args.chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
