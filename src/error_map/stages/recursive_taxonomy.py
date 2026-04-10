import json
import os
from typing import Any, List, Dict, Union, Set
from error_map.stages.error_classification import classify_errors
from error_map.stages.taxonomy_construction import construct_taxonomy
from error_map.stages.taxonomy_population import populate_taxonomy
from error_map.utils.constants import TaxonomyParams
from error_map.utils.taxonomy_tree import TaxonomyNode, TaxonomyTree
from ..utils.cache import cached
from ..core.config import Config
from ..inference import InferenceClient
import math
import asyncio


def _load_existing_taxonomy(taxonomy_path: str) -> Dict:
    """Load a saved taxonomy JSON tree."""
    with open(taxonomy_path) as f:
        tree = json.load(f)
    print(f"Loaded existing taxonomy tree from {taxonomy_path}")
    return tree

def _extract_taxonomy_for_node(tree: Dict, depth: int, node_name: str) -> List[Dict]:
    if not tree:
        return None
        
    target_children = []
    if depth == 0:
        target_children = tree.get("children", [])
    elif depth == 1:
        for child in tree.get("children", []):
            if child["name"] == node_name:
                target_children = child.get("children", [])
                break
                
    clusters = []
    for i, child in enumerate(target_children):
        # Skip leaf nodes (actual error items) which have error_title in info
        if "error_title" in child.get("info", {}):
            continue
            
        desc = child.get("info", {}).get("description", "")
        clusters.append({
            "id": i + 1, 
            "name": child["name"], 
            "description": desc
        })

    if not clusters:
        return None
        
    print(f"Reusing taxonomy for node '{node_name}' (depth {depth}) with {len(clusters)} categories: {[c['name'] for c in clusters]}")
    return [{"judge_response": json.dumps({"clusters": clusters})}]


async def _run_taxonomy_stages(
    records: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    parent_category_name: str = None,
    rare_freq: float = None,
    existing_taxonomy: List[Dict] = None,
) -> List[Dict]:

    # define curr taxonomy size (max num of clusters)
    fixed_max_clusters = config.taxonomy_params["max_num_clusters"]
    curr_max_clusters = min(fixed_max_clusters, math.ceil(len(records)*0.1))
    curr_taxonomy_params = TaxonomyParams.get_modified_taxonomy_params({"max_num_clusters": curr_max_clusters})

    # in case there is already a parent category, specify it in the prompt
    curr_taxonomy_params["parent_category"] = parent_category_name

    if existing_taxonomy:
        taxonomy = existing_taxonomy
    else:
        taxonomy = await construct_taxonomy(
            error_records=records,
            config=config,
            exp_id=exp_id,
            inference_client=inference_client,
            taxonomy_params=curr_taxonomy_params,
        ) if records else []

    if not taxonomy:
        print("ℹ️ No errors to build taxonomy")
        return

    classification = await classify_errors(
        error_records=records,
        error_taxonomy=taxonomy,
        config=config,
        exp_id=exp_id,
        inference_client=inference_client,
    ) if taxonomy else []

    if not classification:
        print("ℹ️ No taxonomy to run classification")
        return
    populated = await populate_taxonomy(
        error_records=records,
        error_taxonomy=taxonomy,
        error_classify=classification,
        exp_id=exp_id,
        config=config,
        rare_freq=rare_freq,
    )

    return populated


def _calculate_max_clusters(config: Config, num_items: int) -> int:
    fixed_max = config.taxonomy_params["max_num_clusters"]
    return min(fixed_max, math.ceil(num_items * 0.1))


def _get_str_from_params(parent: str, name: str, depth: int) -> str:
    return f"parent={parent}__name={name}__depth={depth}"


def _add_children_to_node(
    items: Union[List[Dict], Dict[str, str]],
    parent_node: TaxonomyNode,
    taxonomy_tree: TaxonomyTree,
    depth: int,
):
    if isinstance(items, list):  # Records case
        for item in items:
            if not item:
                continue
            name = item.get("error_title", "Unknown")
            # Use example_id to make leaf IDs unique — prevents deduplication
            # of instances that share the same error_title
            instance_id = item.get("example_id", id(item))
            child = TaxonomyNode(
                id=_get_str_from_params(parent=parent_node.name, name=name, depth=depth) + f"__instance={instance_id}",
                name=name,
                info=item,
            )
            taxonomy_tree.add_node(parent_node=parent_node, child=child)

    elif isinstance(items, dict):  # Categories case
        for name, description in items.items():
            if not name:
                continue
            child = TaxonomyNode(
                id=_get_str_from_params(parent=parent_node.name, name=name, depth=depth),
                name=name,
                info={"description": description},
            )
            taxonomy_tree.add_node(parent_node=parent_node, child=child)

    print(f"Added {len(parent_node.children)} items under '{parent_node.name}'")


async def _recurse_error_collection(
    records: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    parent_node: TaxonomyNode = None,
    depth: int = 0,
    max_depth: int = 2,
    taxonomy_tree: TaxonomyTree = None,
    rare_freq: float = None,
    existing_taxonomy_tree: Dict = None,
):
    print(f"in recurse, records: {len(records)}")

    parent_node_name = parent_node.name if depth > 0 and parent_node.name else None # avoid using the name of the root node or an empty string
    
    # Extract the applicable taxonomy for this level based on the tree structure
    node_taxonomy = _extract_taxonomy_for_node(existing_taxonomy_tree, depth, parent_node_name)
    
    populated = await _run_taxonomy_stages(records, config, exp_id, inference_client, parent_category_name=parent_node_name, rare_freq=rare_freq, existing_taxonomy=node_taxonomy)
    if not populated:
        return

    categories = {
        record["error_category"]: record["category_description"]
        for record in populated if record["error_category"]
    }

    if len(categories) <= 1:
        _add_children_to_node(records, parent_node, taxonomy_tree, depth)
        return

    _add_children_to_node(categories, parent_node, taxonomy_tree, depth)

    tasks = []
    for category in categories.keys():
        curr_records = [r for r in populated if r["error_category"] == category]
        unique_titles = set(r["error_title"] for r in curr_records)

        if not unique_titles:
            print(f"⚠️ Category '{category}' has no children!")
            continue

        curr_max_clusters = _calculate_max_clusters(config, len(unique_titles))
        category_node_id = _get_str_from_params(parent=parent_node.name, name=category, depth=depth)
        category_node = taxonomy_tree.get_node(id=category_node_id)

        if len(unique_titles) <= 5 or curr_max_clusters <= 1 or depth + 1 > max_depth:
            _add_children_to_node(curr_records, category_node, taxonomy_tree, depth)
        else:
            tasks.append(_recurse_error_collection(
                records=curr_records,
                config=config,
                exp_id=exp_id,
                inference_client=inference_client,
                parent_node=category_node,
                depth=depth + 1,
                max_depth=max_depth,
                taxonomy_tree=taxonomy_tree,
                rare_freq=rare_freq,
                existing_taxonomy_tree=existing_taxonomy_tree,
            ))

    if tasks:
        await asyncio.gather(*tasks)

@cached("construct_taxonomy_recursively", None)
async def construct_taxonomy_recursively(
    records: List[Dict],
    config: Config,
    exp_id: str,
    inference_client: InferenceClient,
    depth: int = 0,
    max_depth: int = 2,
    rare_freq: float = None,
    cols_to_keep: List[str] = None,
    reuse_taxonomy_path: str = None,
) -> List[Dict]:
    root_name = "LLM Errors"
    root = TaxonomyNode(
        id=_get_str_from_params(parent=None, name=root_name, depth=depth),
        name=root_name,
    )
    taxonomy_tree = TaxonomyTree(root)

    existing_taxonomy_tree = _load_existing_taxonomy(reuse_taxonomy_path) if reuse_taxonomy_path else None

    await _recurse_error_collection(
                records=records,
                config=config,
                exp_id=exp_id,
                inference_client=inference_client,
                parent_node=root,
                depth=depth,
                max_depth=max_depth,
                taxonomy_tree=taxonomy_tree,
                existing_taxonomy_tree=existing_taxonomy_tree,
                rare_freq=rare_freq,
            )
    try:
        with open(os.path.join(config.output_dir, "exp_name=construct_taxonomy_recursively__exp_id=" + exp_id + ".json"), "w") as f:
            json.dump(taxonomy_tree.to_dict(), f, indent=2)
    except Exception as e:
        print(e)

    # taxonomy tree to records
    results = taxonomy_tree.get_leaf_node_dicts_with_ancestry(cols_to_keep=cols_to_keep)

    return results