#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    meta_subdir: str
    signal_subdir: tuple[str, ...]
    case_col: str
    fs: int


DATASET_SPECS = (
    DatasetSpec(
        name="VitalDB_g",
        meta_subdir="VitalDB_g",
        signal_subdir=("pretrain_signals", "PulseDB_Vital", "generated"),
        case_col="VitalDBid",
        fs=500,
    ),
    DatasetSpec(
        name="MESA_g",
        meta_subdir="MESA_g",
        signal_subdir=("pretrain_signals", "MESA", "generated"),
        case_col="mesaid",
        fs=256,
    ),
    DatasetSpec(
        name="MIMIC_g",
        meta_subdir="MIMIC_g",
        signal_subdir=("pretrain_signals", "PulseDB_MIMIC", "generated"),
        case_col="mesaid",
        fs=125,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the released MorphoPulse synthetic g split to one flattened NPY."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Repository datasets root containing pretrain/ and pretrain_signals/.",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        required=True,
        help="Destination .npy path.",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=None,
        help="Optional flattened metadata CSV path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV chunk size for metadata streaming.",
    )
    return parser.parse_args()


def filter_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["svri"] > 0) & (df["svri"] < 2)]
    df = df[(df["ipa"] > -10) & (df["ipa"] < 10)]
    df = df[(df["skewness"] > -3) & (df["skewness"] < 3)]
    return df


def iter_filtered_chunks(
    data_root: Path,
    spec: DatasetSpec,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    meta_path = data_root / "pretrain" / spec.meta_subdir / "train_clean.csv"
    signal_root = data_root.joinpath(*spec.signal_subdir)
    usecols = [spec.case_col, "segments", "svri", "skewness", "ipa"]
    for chunk in pd.read_csv(meta_path, usecols=usecols, chunksize=chunksize):
        chunk = filter_chunk(chunk)
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={spec.case_col: "case_id"}).copy()
        chunk["case_id"] = chunk["case_id"].map(lambda x: str(x).zfill(4))
        chunk["source_dataset"] = spec.name
        chunk["fs"] = spec.fs
        chunk["path"] = str(signal_root)
        yield chunk


def count_rows(data_root: Path, chunksize: int) -> int:
    total = 0
    for spec in DATASET_SPECS:
        for chunk in iter_filtered_chunks(data_root, spec, chunksize):
            total += len(chunk)
    return total


def load_signal(path: Path) -> np.ndarray:
    sample = joblib.load(path)
    key = "ppg" if "ppg" in sample else "signal"
    signal = sample[key]
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal at {path}, got shape {signal.shape}")
    return signal.astype(np.float32, copy=False)


def discover_layout(data_root: Path, chunksize: int) -> tuple[np.dtype, int]:
    for spec in DATASET_SPECS:
        for chunk in iter_filtered_chunks(data_root, spec, chunksize):
            row = chunk.iloc[0]
            signal_path = Path(row["path"]) / row["case_id"] / row["segments"]
            signal = load_signal(signal_path)
            return signal.dtype, int(signal.shape[0])
    raise RuntimeError("No valid rows found in the synthetic g split.")


def init_meta_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "global_index",
                "source_dataset",
                "case_id",
                "segments",
                "svri",
                "skewness",
                "ipa",
                "fs",
            ]
        )


def append_meta_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)

    total_rows = count_rows(args.data_root, args.chunksize)
    dtype, signal_length = discover_layout(args.data_root, args.chunksize)

    mmap = np.lib.format.open_memmap(
        args.output_npy,
        mode="w+",
        dtype=dtype,
        shape=(total_rows, signal_length),
    )

    if args.output_meta is not None:
        init_meta_csv(args.output_meta)

    progress = tqdm(total=total_rows, desc="Exporting MorphoPulse g synthetic split")
    global_index = 0

    for spec in DATASET_SPECS:
        for chunk in iter_filtered_chunks(args.data_root, spec, args.chunksize):
            meta_rows: list[list[object]] = []
            for row in chunk.itertuples(index=False):
                signal_path = Path(row.path) / row.case_id / row.segments
                signal = load_signal(signal_path)
                if signal.shape[0] != signal_length:
                    raise ValueError(
                        f"Signal length mismatch at {signal_path}: "
                        f"expected {signal_length}, got {signal.shape[0]}"
                    )
                mmap[global_index] = signal
                if args.output_meta is not None:
                    meta_rows.append(
                        [
                            global_index,
                            row.source_dataset,
                            row.case_id,
                            row.segments,
                            row.svri,
                            row.skewness,
                            row.ipa,
                            row.fs,
                        ]
                    )
                global_index += 1
                progress.update(1)
            if args.output_meta is not None and meta_rows:
                append_meta_csv(args.output_meta, meta_rows)

    progress.close()
    mmap.flush()

    if global_index != total_rows:
        raise RuntimeError(f"Exported {global_index} rows, expected {total_rows}")

    gib = (total_rows * signal_length * np.dtype(dtype).itemsize) / (1024 ** 3)
    print(
        f"Wrote {total_rows} segments with shape ({signal_length},) and dtype {dtype} "
        f"to {args.output_npy} (~{gib:.2f} GiB)."
    )


if __name__ == "__main__":
    main()
