#!/usr/bin/env python3
"""
Strip a taxonomy JSON down to a template tree (categories only, no leaf instances).

Usage:
    python make_template_tree.py taxonomy.json
    python make_template_tree.py taxonomy.json --output template.json
"""

import argparse
import json
from pathlib import Path


def strip_leaves(node: dict, depth: int = 0, max_depth: int = None) -> dict | None:
    """Return a copy of node with leaf instances removed. Returns None if node is a leaf."""
    if "error_title" in node.get("info", {}):
        return None  # leaf instance — drop it

    if max_depth is not None and depth >= max_depth:
        return {
            "name": node["name"],
            "info": {"description": node.get("info", {}).get("description", "")},
            "children": [],
        }

    pruned_children = []
    for child in node.get("children", []):
        result = strip_leaves(child, depth + 1, max_depth)
        if result is not None:
            pruned_children.append(result)

    return {
        "name": node["name"],
        "info": {"description": node.get("info", {}).get("description", "")},
        "children": pruned_children,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert a taxonomy JSON to an empty template tree.")
    parser.add_argument("input", help="Path to the taxonomy JSON file")
    parser.add_argument("--output", help="Output path (default: <input>_template.json)")
    parser.add_argument("--first-level", action="store_true", help="Only keep top-level categories (no sub-categories)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_template.json")

    with open(input_path) as f:
        tree = json.load(f)

    max_depth = 1 if args.first_level else None
    template = strip_leaves(tree, max_depth=max_depth)

    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Template written to {output_path}")

    # Print summary
    def count_categories(node, depth=0):
        if not node:
            return
        if depth > 0:
            print(f"{'  ' * depth}- {node['name']}")
        else:
            print(node['name'])
        for child in node.get("children", []):
            count_categories(child, depth + 1)

    count_categories(template)


if __name__ == "__main__":
    main()
