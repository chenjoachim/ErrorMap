#!/usr/bin/env python3
"""
Convert Open ASR benchmark JSON results to ErrorMap CSV format.

Usage:
    python convert_asr.py results.json --model whisper-large-v3
    python convert_asr.py results.json --model whisper-large-v3 --output data/my_benchmark.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path


def convert(input_path: str, output_path: str, model_name: str, scorer_key: str | None = None) -> int:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        scores = item.get("scores", {})

        # Pick the specified scorer or fall back to the first one
        if scorer_key:
            scorer = scores.get(scorer_key, {})
        else:
            scorer = next(iter(scores.values()), {}) if scores else {}

        wer = scorer.get("metadata", {}).get("wer", 0.0)
        score = round(1.0 - wer / 100.0, 6)

        meta = item.get("metadata", {})
        language = meta.get("language", "en")
        dataset = meta.get("dataset", "unknown")
        split = meta.get("split", "test")

        # The judge model will use input_text to understand the task context.
        # Since there is no audio text, we encode domain metadata here.
        input_text = f"Transcribe {language} speech from the {dataset} {split} set."

        rows.append({
            "example_id": item["id"],
            "model": model_name,
            "input_text": input_text,
            "output_text": scorer.get("answer", ""),
            "score": score,
            "correct_answer": item.get("target", ""),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["example_id", "model", "input_text", "output_text", "score", "correct_answer"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {len(rows)} records → {output_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert Open ASR JSON results to ErrorMap CSV.")
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("--model", required=True, help="ASR model name (e.g. whisper-large-v3)")
    parser.add_argument(
        "--output",
        help="Output CSV path (default: data/<input_stem>.csv)",
    )
    parser.add_argument(
        "--scorer",
        help="Scorer key to read from scores dict (default: first key found)",
    )
    args = parser.parse_args()

    output = args.output or f"data/{Path(args.input).stem}.csv"
    convert(args.input, output, args.model, args.scorer)


if __name__ == "__main__":
    main()
