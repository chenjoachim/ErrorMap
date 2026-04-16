#!/usr/bin/env python3
"""
Convert batches of Inspect AI .eval files (Zip format) to ErrorMap CSV format.
Extracts summaries.json from each .eval file and writes rows to output CSVs.

Usage:
    python convert_all_evals.py
    python convert_all_evals.py --evals-dir higgs_evals/higgs-audio-m3__erik-v3 --output-dir data/higgs-audio-m3__erik-v3
"""

import argparse
import json
import os
import zipfile
import csv
from pathlib import Path
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer

_english_normalizer = EnglishTextNormalizer()
_basic_normalizer = BasicTextNormalizer()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Inspect AI .eval files to ErrorMap CSV format.")
    parser.add_argument(
        "--evals-dir",
        default="higgs_evals",
        help="Directory containing .eval files (default: higgs_evals)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write output CSVs (default: data)",
    )
    return parser.parse_args()


def normalize_text(text: str, language: str = "en") -> str:
    """Normalize ASR text using Whisper's normalizers.

    Uses EnglishTextNormalizer for English (matches open_asr leaderboard),
    and BasicTextNormalizer for all other languages.
    """
    if language == "en" or language == "en-us":
        return _english_normalizer(text)
    return _basic_normalizer(text)


def process_eval_file(eval_path, all_dataset_rows):
    filename = os.path.basename(eval_path)
    
    # Example format: open_asr_ami__chk65k-v3_2026-03-31T...eval
    parts = filename.split("__")
    if len(parts) >= 2:
        benchmark = parts[0]
        # The remainder is something like: chk65k-v3_2026-03-31T22-26-12...
        # We split by '_2026-' to effectively extract the model tag on the left.
        model_part = parts[1]
        model_name = model_part.split("_2026-")[0]
    else:
        benchmark = "unknown_dataset"
        model_name = "unknown_model"

    # Inspect AI .eval files are Zip files.
    # We load `summaries.json` straight from the archive.
    try:
        with zipfile.ZipFile(eval_path, 'r') as z:
            with z.open('summaries.json') as f:
                data = json.load(f)
    except Exception as e:
        print(f"Failed to read summaries.json from {filename}: {e}")
        return

    if benchmark not in all_dataset_rows:
        all_dataset_rows[benchmark] = []

    for item in data:
        scores = item.get("scores", {})
        # Grab first scorer metric available
        scorer = next(iter(scores.values()), {}) if scores else {}

        metadata = scorer.get("metadata", {})
        value = scorer.get("value", {})
        
        # We attempt to compute word error rate (wer)
        wer = 0.0
        if isinstance(value, (float, int)):
            if "wer" in metadata:
                wer = float(metadata["wer"])
            else:
                wer = float(value)
        elif isinstance(value, dict):
            if "distance" in value and "ref_length" in value:
                distance = value.get("distance", 0)
                ref_length = value.get("ref_length", 1) # avoid DivByZero
                if ref_length > 0:
                    wer = (float(distance) / float(ref_length)) * 100.0
                else:
                    wer = 100.0
        
        # Score transforms WER back to a standard 0 to 1 scale (higher is better)
        score_val = round(1.0 - (wer / 100.0), 6)

        item_meta = item.get("metadata", {})
        language = item_meta.get("language", "en")
        dataset = item_meta.get("dataset", benchmark)
        split = item_meta.get("split", "test")

        # Set up a prompt description
        input_text = "You are an automatic speech recognition (ASR) system. [audio]"

        all_dataset_rows[benchmark].append({
            "example_id": item.get("id", ""),
            "model": model_name,
            "input_text": input_text,
            "output_text": normalize_text(scorer.get("answer", ""), language),
            "score": score_val,
            "correct_answer": normalize_text(item.get("target", ""), language),
            "ref_length": ref_length if 'ref_length' in locals() else len(normalize_text(item.get("target", ""), language).split()),
        })


def main():
    args = parse_args()
    evals_dir = args.evals_dir
    output_dir = args.output_dir

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_dataset_rows = {}

    eval_files = list(Path(evals_dir).glob("*.eval"))
    print(f"Found {len(eval_files)} .eval files in {evals_dir}.")
    
    if not eval_files:
        print("Nothing to process.")
        return

    for eval_file in eval_files:
        print(f"Processing {eval_file.name}...")
        process_eval_file(eval_file, all_dataset_rows)

    for benchmark, rows in all_dataset_rows.items():
        output_csv = Path(output_dir) / f"{benchmark}.csv"
        
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["example_id", "model", "input_text", "output_text", "score", "correct_answer", "ref_length"],
            )
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"➞ Saved {len(rows)} combined records to {output_csv}")

    print("Success! All metrics compiled to CSV format.")


if __name__ == "__main__":
    main()
