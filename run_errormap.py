#!/usr/bin/env python3
"""
Run ErrorMap error analysis across multiple converted ASR benchmark CSVs and models.

Reads OPENAI_API_KEY from .env in the current directory.

Usage:
    python run_errormap.py --dataset open_asr_ami open_asr_ted --model whisper-large-v3
    python run_errormap.py --dataset open_asr_ami fleurs_asr_en --model whisper-large-v3 qwen-audio --ratio 0.5 --threshold 0.8 --max-per-dataset 50
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from error_map import ErrorMap


# def print_taxonomy_summary(output_dir: str, exp_id: str, n_examples: int = 2) -> None:
#     json_path = Path(output_dir) / f"exp_name=construct_taxonomy_recursively__exp_id={exp_id}.json"
#     single_error_path = Path(output_dir) / f"exp_name=single_error__exp_id={exp_id}.csv"

#     if not json_path.exists():
#         print(f"Taxonomy JSON not found at {json_path}")
#         return

#     with open(json_path) as f:
#         tree = json.load(f)

#     # Build error_title → [instances] from the single_error CSV (one row per instance)
#     title_to_instances: dict = {}
#     if single_error_path.exists():
#         df = pd.read_csv(single_error_path)
#         for _, row in df.iterrows():
#             try:
#                 jr = json.loads(row.get("judge_response") or "{}")
#                 title = jr.get("final_answer", {}).get("error_title", "").strip()
#                 if title:
#                     title_to_instances.setdefault(title, []).append(row.to_dict())
#             except (json.JSONDecodeError, TypeError):
#                 pass

#     total = sum(len(v) for v in title_to_instances.values())

#     def instance_count(node: dict) -> int:
#         if not node["children"]:
#             return len(title_to_instances.get(node["name"], []))
#         return sum(instance_count(c) for c in node["children"])

#     def print_examples(instances: list) -> None:
#         for row in instances[:n_examples]:
#             summary = str(row.get("error_summary", "") or "").strip()
#             hyp = str(row.get("output_text", "") or "").strip()
#             ref = str(row.get("correct_answer", "") or "").strip()
#             if summary:
#                 print(f'         e.g. "{summary}"')
#             if ref and hyp:
#                 print(f"              ref: {ref[:80]}")
#                 print(f"              hyp: {hyp[:80]}")

#     def print_node(node: dict, depth: int) -> None:
#         count = instance_count(node)
#         pct = f"{count / total * 100:.1f}%" if total else "—"
#         indent = "    " * depth
#         marker = "├─ " if depth > 0 else ""
#         print(f"\n{indent}{marker}{node['name']}  [{count} instances, {pct}]")

#         if not node["children"]:
#             print_examples(title_to_instances.get(node["name"], []))
#         else:
#             for child in node["children"]:
#                 print_node(child, depth + 1)

#     print(f"\n{'─'*60}")
#     print(f"  Taxonomy Summary  ({total} instances)")
#     print(f"{'─'*60}")

#     for top in tree.get("children", []):
#         print_node(top, depth=1)

#     print(f"\n{'─'*60}\n")

load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="Run ErrorMap on an ASR benchmark CSV.")
    parser.add_argument(
        "--dataset",
        required=True,
        nargs="+",
        help="Dataset name(s) — must match the CSV filename in data/ (without .csv)",
    )
    parser.add_argument("--model", required=True, nargs="+", help="ASR model name(s) to analyze")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Success threshold for score (default: 0.85 → WER > 15%% is an error)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of errors to sample for analysis (default: 1.0 = all)",
    )
    parser.add_argument("--max-workers", type=int, default=20, help="Concurrent inference workers")
    parser.add_argument("--max-per-dataset", type=int, help="Maximum number of errors to sample per dataset")
    parser.add_argument("--exp-id", help="Custom experiment ID")
    parser.add_argument("--asr", action="store_true", default=False, help="Use ASR-specific error analysis prompt")
    parser.add_argument("--data-path", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--output-dir", default="output", help="Path to outputs directory (default: output)")
    parser.add_argument("--reuse-taxonomy", help="Path to existing taxonomy JSON to reuse top-level categories")
    args = parser.parse_args()

    if "all" in args.dataset:
        args.dataset = [f.stem for f in Path("data").glob("*.csv")]
        if not args.dataset:
            raise SystemExit("No CSV files found in data/ directory.")

    if "all" in args.model:
        args.model = None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found. Set it in your .env file.")

    error_map = ErrorMap(
        inference_type="litellm",
        litellm_config={
            "model": "gpt-5-mini",
            "api_key": api_key,
            "max_tokens": 4000,
            "reasoning_effort": "minimal",
        },
        data_path=args.data_path,
        datasets=args.dataset,
        dataset_params={
            ds: {"success_threshold": args.threshold} for ds in args.dataset
        },
        models=args.model,
        ratio=args.ratio,
        max_workers=args.max_workers,
        exp_id=args.exp_id,
        output_dir=args.output_dir,
        asr=args.asr,
        reuse_taxonomy_path=args.reuse_taxonomy,
        max_per_dataset=args.max_per_dataset,
    )

    result = await error_map.run()
    print(f"\nDone — experiment: {result['exp_id']}")
    print(f"Total records: {result['total_records']}, errors analyzed: {result['error_records']}")
    print(f"Results saved to: output/")
    # print_taxonomy_summary(args.output_dir, result["exp_id"])


if __name__ == "__main__":
    asyncio.run(main())
