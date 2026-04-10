
import asyncio
import csv
import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from error_map.stages.taxonomy_construction import _extract_description
from ..utils.cache import cached
from ..core.config import Config
from ..inference import InferenceClient
from collections import Counter
from tqdm.asyncio import tqdm_asyncio


def get_last_exsiting_taxonomy(error_taxonomy: List[Dict]) -> Dict:
    taxonomy_dict = None
    for i in range(len(error_taxonomy) - 1, -1, -1):
        taxonomy_json = error_taxonomy[i]["judge_response"]
        try:
            taxonomy_dict = json.loads(taxonomy_json)
            num_categories = len(taxonomy_dict["clusters"])
            if i == len(error_taxonomy) - 1:
                print(f"Using reviewed taxonomy, with {num_categories} categories..")
            else:
                print(f"Reviewed taxonomy wasn't found! Using taxonomy from iter: {i+1}/{len(error_taxonomy)}, with {num_categories} categories..")
            break
        except:
            continue
    return taxonomy_dict


async def classify_batch(record_batch: List[Dict], taxonomy: Dict, inference_client: InferenceClient) -> Dict:
    """Classify a batch of {title, summary} record dicts and return index→category mapping."""
    data = [{"title": r["title"], "summary": r["summary"]} for r in record_batch]
    template_vars = {
        "data": data,
        "taxonomy": taxonomy,
    }

    try:
        result = await inference_client.infer("classify_errors.j2", template_vars, schema_name="classify_errors_schema.json")
        categories = json.loads(result["content"]).get("classified_errors", [])
        return {
            "prompt": result["prompt"],
            "judge_model": result["model"],
            "judge_response": result["content"],
            "template_used": result["template"],
            "inference_success": result["success"],
            "full_response": result["full_response"],
            # Attach per-record results for downstream mapping
            "record_categories": [
                {"record_id": r["record_id"], "category": categories[i] if i < len(categories) else "Other"}
                for i, r in enumerate(record_batch)
            ],
        }
    except Exception as e:
        return {
            "error": str(e),
            "record_categories": [{"record_id": r["record_id"], "category": "Other"} for r in record_batch],
        }


async def classify_errors(
    error_records: List[Dict],
    error_taxonomy: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    field: str = "error_title",
) -> List[Dict]:
    print(f"Classifying {len(error_records)} errors to taxonomy...")

    # Extract title + summary for each record individually (no deduplication)
    titles = await asyncio.gather(*[_extract_description(record, "error_title") for record in error_records])
    summaries = await asyncio.gather(*[_extract_description(record, "error_summary") for record in error_records])

    record_inputs = [
        {"record_id": i, "title": titles[i] or "", "summary": summaries[i] or ""}
        for i in range(len(error_records))
        if titles[i]
    ]

    # use final existing taxonomy
    taxonomy = get_last_exsiting_taxonomy(error_taxonomy)

    # send batches of 5 records to be classified
    batch_size = config.taxonomy_params.get("classify_batch_size", 5)
    batches = [record_inputs[i:i + batch_size] for i in range(0, len(record_inputs), batch_size)]

    # classify batches in parallel
    return await tqdm_asyncio.gather(*[classify_batch(batch, taxonomy, inference_client) for batch in batches])
