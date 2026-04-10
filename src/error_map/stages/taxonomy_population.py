

from ast import Dict
import asyncio
from collections import defaultdict
from typing import List
from error_map.core.config import Config
from error_map.stages.error_classification import get_last_exsiting_taxonomy
from error_map.stages.taxonomy_construction import _extract_description
from error_map.utils.cache import cached
import json
from typing import Any, Dict, List
from collections import defaultdict
from typing import List, Dict
import json
import pandas as pd

def _norm(text: str) -> str:
    return text.strip().lower()

def _map_error_to_category(results: List[Dict], categories: Dict) -> Dict:
    norm_categories = {_norm(category): category for category in categories.keys()}
    id2category = {}

    for result in results:
        for entry in result.get("record_categories", []):
            try:
                record_id = entry.get("record_id")
                err_cat = (entry.get("category") or "").strip()

                if record_id is None or not err_cat:
                    print(f"Error: Missing record_id or category in entry: {entry}")
                    continue

                norm_err_cat = _norm(err_cat)
                if norm_err_cat not in norm_categories:
                    print(f"Error category '{err_cat}' doesn't exist! (record_id={record_id}). Applying 'Other' category instead.")
                    norm_err_cat = _norm("Other")

                id2category[record_id] = norm_categories[norm_err_cat]

            except Exception as e:
                print(f"Unexpected error while processing entry: {e}")
                continue

    return id2category


def _norm(s: Any) -> str:
    return ("" if s is None else str(s)).strip().lower()


async def _merge_records_with_categories(record: Dict, record_index: int, error_title: str, error_summary: str, id2category: Dict, categories: Dict) -> Dict:
    cat = ""
    cat_desc = ""
    if error_title:
        if record_index not in id2category:
            print(f"Error label haven't been assigned with a category! record_index: {record_index}, error text: {error_title}. Assigning 'Other' category instead.")
        cat = id2category.get(record_index, "Other")
        cat_desc = categories.get(cat, "")

    return {
        **record,
        "error_title": error_title,
        "error_summary": error_summary,
        "error_category": cat,
        "category_description": cat_desc,
    }


def _replace_rare_categories_with_other(records: List[Dict], rare_freq: float = 0.0) -> List[Dict]:
    if rare_freq is None or rare_freq == 0:
        return records
    
    df = pd.DataFrame(records)
    category_counts = df['error_category'].value_counts(normalize=True) * 100
    list_of_tuples = list(category_counts.items())
    rare_categories = [cat for cat, freq in list_of_tuples if freq < rare_freq * 100]
    
    mask = df['error_category'].isin(rare_categories)
    df.loc[mask, 'error_category'] = 'Other'
    df.loc[mask, 'category_description'] = ''

    print(f"Found {len(rare_categories)} rare categories (<2%), {mask.sum()} errors were classified to 'Other'.")
    return df.to_dict('records')


async def populate_taxonomy(
    error_records: List[Dict],
    error_taxonomy: List[Dict],
    error_classify: List[Dict],
    exp_id: str,
    config: Config,
    rare_freq: float,
    ) -> List[Dict]:
    print("Creating final result: populated taxonomy...")
    
    # aggregate final categories
    categories = get_last_exsiting_taxonomy(error_taxonomy)
    if not categories:
        raise Exception("Unable to extract categories from the data. Please try running the process again.")
    
    categories = categories["clusters"]
    categories = {item["name"]: item["description"] for item in categories if "name" in item and "description" in item}
    categories["Other"] = ""

    # error to category map
    id2category = _map_error_to_category(error_classify, categories)
    
    # extract record descriptions 
    error_titles = await asyncio.gather(*[_extract_description(record, "error_title") for record in error_records])
    error_summaries = await asyncio.gather(*[_extract_description(record, "error_summary") for record in error_records])
    
    # for each error record add error description and error category fields
    result = await asyncio.gather(*[_merge_records_with_categories(record, ind, error_titles[ind], error_summaries[ind], id2category, categories) for ind, record in enumerate(error_records)])

    # replace rare categories with other default category
    result_after_rare_categories_drop = _replace_rare_categories_with_other(result, rare_freq=rare_freq)

    return result_after_rare_categories_drop
