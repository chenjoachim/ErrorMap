#!/usr/bin/env python3
"""
Generate a summary report of error type frequencies and representative examples
from a categorized ErrorMap output CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import json

def parse_category_name(cat_str: str) -> str:
    try:
        return json.loads(cat_str).get("name", cat_str)
    except Exception:
        return str(cat_str)

def load_classified_data(csv_path: str) -> pd.DataFrame:
    """Load the output CSV from the taxonomy stage."""
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    return pd.read_csv(path)


def extract_exp_id(csv_path: Path) -> str | None:
    """Extract exp_id from a filename like exp_name=...___exp_id=<id>.csv."""
    stem = csv_path.stem  # strip .csv
    marker = "__exp_id="
    idx = stem.find(marker)
    if idx == -1:
        return None
    return stem[idx + len(marker):]


def enrich_with_targets(df: pd.DataFrame, exp_id: str, output_dir: Path) -> pd.DataFrame:
    """Join correct_answer from the data_preparation stage output onto df."""
    prep_path = output_dir / f"exp_name=data_preparation__exp_id={exp_id}.csv"
    if not prep_path.exists():
        print(f"Warning: data_preparation file not found at {prep_path}", file=sys.stderr)
        return df
    prep_df = pd.read_csv(prep_path, usecols=lambda c: c in {"example_id", "model", "correct_answer"})
    return df.merge(prep_df, on=["example_id", "model"], how="left")


def enrich_with_ref_length(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Join ref_length from all CSVs in data/ (written by convert_all_evals.py)."""
    if "ref_length" in df.columns:
        return df

    pieces = []
    for src in data_dir.glob("*.csv"):
        src_df = pd.read_csv(src, usecols=lambda c: c in {"example_id", "model", "ref_length"})
        if "ref_length" not in src_df.columns:
            continue
        pieces.append(src_df)

    if not pieces:
        return df

    ref_df = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["example_id", "model"])
    return df.merge(ref_df, on=["example_id", "model"], how="left")


def _corpus_wer(subset: pd.DataFrame) -> float:
    """Compute corpus-level WER using ref_length weighting when available, else mean WER."""
    wer = (1.0 - subset["score"]) * 100.0
    if "ref_length" in subset.columns:
        ref_len = pd.to_numeric(subset["ref_length"], errors="coerce")
        valid = ref_len.notna() & (ref_len > 0)
        if valid.any():
            errors = (wer[valid] * ref_len[valid] / 100.0).sum()
            return 100.0 * errors / ref_len[valid].sum()
    return wer.mean()


def load_full_benchmark_data(data_dir: Path, models: list[str]) -> pd.DataFrame | None:
    """Load all samples from data/ CSVs, filtered to the given models, with dataset column added."""
    pieces = []
    for src in data_dir.glob("*.csv"):
        src_df = pd.read_csv(src, usecols=lambda c: c in {"model", "score", "ref_length"})
        src_df = src_df[src_df["model"].isin(models)]
        src_df["dataset"] = src.stem
        pieces.append(src_df)
    if not pieces:
        return None
    return pd.concat(pieces, ignore_index=True)


def _wer_row(dataset: str, group: pd.DataFrame, has_ref_length: bool) -> str:
    """Format a single benchmark row for the master WER table."""
    overall_wer = f"{_corpus_wer(group):.2f}%"
    critical_pct = f"{(group['score'] <= 0.85).mean() * 100:.1f}%"
    n = f"{len(group):,}"

    if has_ref_length:
        short  = group[group["ref_length"] <= 5]
        medium = group[(group["ref_length"] >= 6) & (group["ref_length"] <= 50)]
        long_  = group[group["ref_length"] > 50]
        short_wer  = f"{_corpus_wer(short):.2f}%"  if not short.empty  else "N/A"
        medium_wer = f"{_corpus_wer(medium):.2f}%" if not medium.empty else "N/A"
        long_wer   = f"{_corpus_wer(long_):.2f}%"  if not long_.empty  else "N/A"
    else:
        short_wer = medium_wer = long_wer = "N/A"

    return f"| {dataset} | {n} | {overall_wer} | {short_wer} | {medium_wer} | {long_wer} | {critical_pct} |"


def _print_wer_table(full_df: pd.DataFrame, datasets: list[str], title: str) -> None:
    subset = full_df[full_df["dataset"].isin(datasets)]
    if subset.empty:
        return
    has_ref_length = "ref_length" in subset.columns

    print(f"### {title}")
    print()
    print("| Benchmark | Total Utterances | Overall WER | Short WER | Medium WER | Long WER | Critical Error % |")
    print("|---|---|---|---|---|---|---|")
    for dataset, group in subset.groupby("dataset"):
        print(_wer_row(dataset, group, has_ref_length))
    print()


def print_wer_by_benchmark(full_df: pd.DataFrame) -> None:
    """Print master WER table split into FLEURS and Open ASR groups."""
    if full_df is None or "score" not in full_df.columns:
        return

    all_datasets = full_df["dataset"].unique().tolist()
    fleurs   = sorted(d for d in all_datasets if d.startswith("fleurs_"))
    open_asr = sorted(d for d in all_datasets if d.startswith("open_asr_"))
    other    = sorted(d for d in all_datasets if d not in fleurs and d not in open_asr)

    print("> **Definitions**")
    print("> - **Short**: utterances with ≤ 5 reference words")
    print("> - **Medium**: utterances with 6–50 reference words")
    print("> - **Long**: utterances with > 50 reference words")
    print("> - **Critical Error %**: share of utterances with WER ≥ 15% (score ≤ 0.85) — the population that enters ErrorMap analysis")
    print()
    _print_wer_table(full_df, fleurs,   "WER — FLEURS (Multilingual)")
    _print_wer_table(full_df, open_asr, "WER — Open ASR (English)")
    if other:
        _print_wer_table(full_df, other, "WER — Other")


def print_overall_distribution(df: pd.DataFrame, category_col: str = "category_depth_0") -> None:
    """Print the overall distribution of error categories across models."""
    if category_col not in df.columns:
        print(f"Column '{category_col}' not found. Cannot print error distribution.", file=sys.stderr)
        return

    print("### Overall Error Distribution")
    print()
    
    models = df["model"].unique()
    
    # Calculate percentage distribution per model
    dist_dfs = []
    for model in models:
        model_df = df[df["model"] == model]
        counts = model_df[category_col].value_counts(normalize=True) * 100
        counts.name = model
        dist_dfs.append(counts)
    
    # Combine into a single table
    dist_df = pd.concat(dist_dfs, axis=1).fillna(0)
    
    # Sort by total mass across models
    dist_df["Total"] = dist_df.sum(axis=1)
    dist_df = dist_df.sort_values(by="Total", ascending=False).drop(columns=["Total"])
    
    # Format as markdown table
    headers = ["Error Category"] + list(dist_df.columns)
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---'] * len(headers))}|")
    
    for category, row in dist_df.iterrows():
        cat_name = parse_category_name(category)
        row_strs = [cat_name] + [f"{val:.1f}%" for val in row]
        print(f"| {' | '.join(row_strs)} |")
    print()


def print_benchmark_breakdown(df: pd.DataFrame, category_col: str = "category_depth_0") -> None:
    """Print top 3 error categories per benchmark."""
    if category_col not in df.columns or "dataset" not in df.columns:
        return

    print("### Per-Benchmark Top 3 Error Categories")
    print()
    print("| Benchmark | #1 | #2 | #3 |")
    print("|---|---|---|---|")
    for dataset, group in df.groupby("dataset"):
        top3 = group[category_col].value_counts(normalize=True).head(3) * 100
        cells = [f"{parse_category_name(cat)} ({pct:.1f}%)" for cat, pct in top3.items()]
        while len(cells) < 3:
            cells.append("—")
        print(f"| {dataset} | {cells[0]} | {cells[1]} | {cells[2]} |")
    print()


def print_representative_examples(df: pd.DataFrame, category_col: str = "category_depth_0", n: int = 1) -> None:
    """Print representative error instances for each category."""
    if category_col not in df.columns:
        return

    print("### Representative Examples per Category")
    print()
    
    categories = df[category_col].value_counts().index
    
    for category in categories:
        cat_name = parse_category_name(category)
        print(f"#### {cat_name}")
        cat_df = df[df[category_col] == category]
        sample = cat_df.sample(min(n, len(cat_df)))
        
        for _, row in sample.iterrows():
            print(f"- **Benchmark:** {row.get('dataset', '?')} | **Model:** {row.get('model', '?')}")
            print(f"- **Error Title:** {row.get('error_title', '')}")
            print(f"- **Summary:** {row.get('error_summary', '')}")
            target = row.get('correct_answer')
            prediction = row.get('output_text')
            if pd.notna(target):
                print(f"  - *Target:* {str(target)[:200]}")
            if pd.notna(prediction):
                print(f"  - *Prediction:* {str(prediction)[:200]}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Generate ErrorMap report from categorized CSV.")
    parser.add_argument("csv_path", help="Path to the categorized output CSV (e.g., from construct_taxonomy_recursively).")
    parser.add_argument("--category-col", default="category_depth_0", help="Column containing the primary error category.")
    parser.add_argument("--examples", type=int, default=1, help="Number of representative examples to show per category.")

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    df = load_classified_data(args.csv_path)
    exp_id = extract_exp_id(csv_path)
    if exp_id is not None:
        df = enrich_with_targets(df, exp_id, csv_path.parent)
    df = enrich_with_ref_length(df, Path("data"))
    print(f"Loaded {len(df)} categorized errors from {args.csv_path}")
    print("=" * 60)
    print()
    
    full_df = load_full_benchmark_data(Path("data"), df["model"].unique().tolist())
    print_wer_by_benchmark(full_df)
    print_overall_distribution(df, category_col=args.category_col)
    print_benchmark_breakdown(df, category_col=args.category_col)
    
    if args.examples > 0:
        print_representative_examples(df, category_col=args.category_col, n=args.examples)


if __name__ == "__main__":
    main()
