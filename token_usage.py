#!/usr/bin/env python3
"""
Calculate total token usage from a single_error CSV file.
Reads the `full_response` column, parses completion_tokens and prompt_tokens
from each ModelResponse string, and prints a summary.

Usage:
    python token_usage.py <path_to_csv>
    python token_usage.py  # uses the default file
"""

import csv
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_CSV = "output/exp_name=single_error__exp_id=higgs_0403_full.csv"

# Regex patterns to extract token counts from the ModelResponse repr string
COMPLETION_RE = re.compile(r"completion_tokens=(\d+)")
PROMPT_RE = re.compile(r"prompt_tokens=(\d+)")


def parse_token_usage(response_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse completion_tokens and prompt_tokens from a ModelResponse repr string.
    Returns (completion_tokens, prompt_tokens) or None if not found.
    """
    comp_match = COMPLETION_RE.search(response_str)
    prompt_match = PROMPT_RE.search(response_str)
    if comp_match and prompt_match:
        return int(comp_match.group(1)), int(prompt_match.group(1))
    return None


def calculate_usage(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    total_completion = 0
    total_prompt = 0
    total_rows = 0
    parsed_rows = 0
    skipped_rows = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "full_response" not in (reader.fieldnames or []):
            print("Error: 'full_response' column not found in CSV.")
            sys.exit(1)

        for row in reader:
            total_rows += 1
            val = row.get("full_response", "")
            if not val or val.strip() in ("", "None", "nan"):
                skipped_rows += 1
                continue

            result = parse_token_usage(val)
            if result is None:
                skipped_rows += 1
                continue

            completion, prompt = result
            total_completion += completion
            total_prompt += prompt
            parsed_rows += 1

    total_tokens = total_completion + total_prompt

    print(f"\nToken Usage Summary — {path.name}")
    print("=" * 50)
    print(f"  Rows processed   : {total_rows:>10,}")
    print(f"  Rows with usage  : {parsed_rows:>10,}")
    print(f"  Rows skipped     : {skipped_rows:>10,}")
    print("-" * 50)
    print(f"  Prompt tokens    : {total_prompt:>10,}")
    print(f"  Completion tokens: {total_completion:>10,}")
    print(f"  Total tokens     : {total_tokens:>10,}")
    print("=" * 50)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    calculate_usage(csv_path)
