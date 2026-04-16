# CLAUDE.md — ErrorMap Project Orientation

> This file is a recap document for AI assistants (Claude, Antigravity, etc.) to quickly orient themselves on the ErrorMap codebase. Update it after significant changes.
> Last updated: 2026-04-16

---

## What Is ErrorMap?

ErrorMap is a Python research tool (academic project by Boson/chenjoachim) that **automatically clusters and categorizes LLM/ASR evaluation errors** into human-readable taxonomies. It takes benchmark CSVs (e.g., ASR model outputs), identifies error cases, analyzes each error with an LLM judge, builds a hierarchical taxonomy of error types, and classifies every error into that taxonomy.

The paper this is based on is **"ErrorMap"** and **"ErrorAtlas"** (citation added Feb 2026 by Shir Ashury-Tahan).

---

## Git Commit History Recap

| Hash | Date | Summary |
|------|------|---------|
| `32b007a` | Initial | Initial repository scaffolding |
| `f891a9f` | Initial | Second init commit (likely repo migration) |
| `3e72ec3` | Initial | Third init commit |
| `704a4cf` | 2026-02-01 | *(by Shir Ashury-Tahan)* Added citations for ErrorMap & ErrorAtlas to README |
| `f4f3363` | 2026-04-09 | **Add ASR support** — new `run_errormap.py` (argparse-driven CLI), `convert_asr.py`, ASR-specific Jinja2 prompt (`single_error_analysis_asr.j2`), updated inference stages for ASR mode, `.gitignore` updates |
| `50b8411` | 2026-04-10 | Added `copy_latest_evals.sh` — bash utility to copy latest evaluation files |
| `b39fe7e` | 2026-04-10 | **Build initial ASR taxonomy** — new `convert_all_evals.py`, major rework of `run_errormap.py` (multi-dataset/multi-model support), `recursive_taxonomy.py` updated for multi-run, `data_preparation.py` improvements |
| `e2b74ed` | 2026-04-10 | **Refactor classify_errors + fix API key leaks** — switched from title-deduplication to **id-based batching** in `classify_errors`, redacted API key in `config.json` output, added `make_template_tree.py`, added `token_usage.py`, updated `classify_errors.j2` and response schema |

---

## Current Workflow

### Entry Point
```
python run_errormap.py \
  --dataset open_asr_ami open_asr_ted \
  --model whisper-large-v3 \
  --asr \
  --threshold 0.8 \
  --ratio 1.0 \
  --max-per-dataset 100 \
  --exp-id my_experiment
```

### Pipeline Stages (in order)

```
run_errormap.py
 └── ErrorMap.run()
      ├── 1. prepare_data()          → loads CSVs from data/, filters by model, samples errors by threshold + ratio
      ├── 2. analyze_single_errors() → per-error LLM analysis using Jinja2 prompts (ASR or generic)
      └── 3. construct_taxonomy_recursively()
               ├── construct_taxonomy()    → LLM clusters error titles into categories
               ├── classify_errors()       → id-based batching; assigns each error to a category
               └── populate_taxonomy()     → builds hierarchical tree; recurses up to max_depth=2
```

Each stage is **cached** to disk (`output/exp_name=<stage>__exp_id=<id>.*`), so interrupted runs can resume without re-invoking the LLM.

### Key Files

| File | Role |
|------|------|
| `run_errormap.py` | CLI entry point; builds `ErrorMap` object and calls `.run()` |
| `src/error_map/__init__.py` | `ErrorMap` class; wires up config, inference client, cached stages |
| `src/error_map/stages/data_preparation.py` | Loads & filters CSV data |
| `src/error_map/stages/single_error.py` | Per-error LLM analysis |
| `src/error_map/stages/recursive_taxonomy.py` | Recursive taxonomy construction (main orchestration) |
| `src/error_map/stages/error_classification.py` | Batched LLM classification into taxonomy categories |
| `src/error_map/stages/taxonomy_construction.py` | LLM call to generate taxonomy clusters |
| `src/error_map/stages/taxonomy_population.py` | Populates tree nodes with classified errors |
| `src/error_map/inference/client.py` | LiteLLM-backed inference client (async) |
| `src/error_map/templates/prompts/single_error_analysis_asr.j2` | Jinja2 prompt for ASR error analysis |
| `src/error_map/templates/prompts/classify_errors.j2` | Prompt for batch error→category classification |
| `src/error_map/templates/response_schemas/classify_errors_schema.json` | JSON schema for classification response |
| `taxonomies/asr_base_v2_template.json` | Pre-built ASR taxonomy template (9 top-level categories) |
| `convert_asr.py` | Converts raw ASR benchmark data into ErrorMap CSV format |
| `convert_all_evals.py` | Batch-converts multiple eval files (including extracting `ref_length`) |
| `copy_latest_evals.sh` | Bash script that copies latest eval CSVs into `data/` |
| `analyze_asr.py` | Standalone script to compare WER between two `.eval` folders. Tracks overall/length-bucketed WER per utterance, and exports regressions to CSV. |
| `report.py` | CLI reporter that takes classified output from the pipeline and renders quantitative distributions and qualitative representative examples. |
| `token_usage.py` | Reads output CSVs and totals prompt/completion token usage |
| `make_template_tree.py` | Utility to generate taxonomy template trees |

### Data Flow

```
higgs_evals/          →  copy_latest_evals.sh  →  data/*.csv
data/*.csv            →  convert_asr.py (if needed)
data/*.csv            →  ErrorMap pipeline
output/               →  exp_name=*.csv / *.json (one file per stage per exp_id)
taxonomies/           →  optional reuse_taxonomy_path flag
```

### ASR-Specific Mode (`--asr` flag)

When `--asr` is set:
- Uses `single_error_analysis_asr.j2` prompt (phonetic/linguistic ASR analysis)
- Prompt analyzes: Ground Truth vs. Hypothesis, identifies error type (e.g., "Homophone Substitution", "Proper Noun Deletion")
- Can optionally pass `--reuse-taxonomy` pointing to `taxonomies/asr_base_v2_template.json` to impose a fixed top-level taxonomy

### The ASR Base Taxonomy (v2)

9 top-level categories defined in `taxonomies/asr_base_v2_template.json`:
1. Out-of-Vocabulary / Proper Noun
2. Phonetic Confusability
3. Word Boundary Segmentation
4. Function-Word Dropout
5. Language Model Bias
6. Numeric and Alphanumeric
7. Speaker / Accent Variability
8. Disfluency and Endpointing
9. Other

### Inference Backend

- Uses **LiteLLM** (`litellm`) as the inference client
- Default model in `run_errormap.py`: `gpt-5-mini` with `reasoning_effort: minimal`, `max_tokens: 4000`
- API key loaded from `.env` file (`OPENAI_API_KEY`)
- API key is **redacted** in saved config JSONs (fixed in latest commit)

---

## Known Patterns & Gotchas

- **Caching**: All three main stages (`prepare_data`, `analyze_single_errors`, `construct_taxonomy_recursively`) are wrapped with `@cached`. If you change logic but use the same `--exp-id`, the old cached output will be reused. Use a new `--exp-id` or delete the relevant `output/` files to re-run.
- **id-based batching** (latest refactor): `classify_errors` now passes `record_id` (index) alongside title/summary per record, preventing deduplication of identical error titles and ensuring every individual error instance is assigned a category.
- **Multi-dataset / multi-model**: Pass multiple `--dataset` and `--model` values; `"all"` is a special keyword.
- **`cols_to_keep`**: Controls which additional columns from the original CSV are preserved in the output records.
- **`rare_freq`**: Merges low-frequency categories into "Other" to avoid long-tail noise in the taxonomy.

---

## Recent Development Focus (as of 2026-04-16)

The most recent development has been around:
1. Extending ErrorMap to support **ASR benchmarks** (previously generic LLM eval)
2. Building and iterating on an **ASR-specific error taxonomy** (`asr_base_v2_template.json`)
3. Refactoring the **classification stage** for correctness (id-based batching)
4. Adding **token usage tracking** (`token_usage.py`)
5. Operational tooling: file copy scripts, batch conversion utilities
6. **Reporting Pipelines**: Standalone error reporting (`report.py`) and model comparative tools (`analyze_asr.py`)

---

## End-to-End Analysis Workflow

To systematically analyze errors or compare models side-by-side, use the following sequence:

### 1. Extract Model Disagreements (For Comparison)

Use the comparison script against raw `.eval` folders to find regressions where Model B performed worse than Model A. Export this directly to ErrorMap CSV format:

```bash
python analyze_asr.py \
  higgs_evals/open_asr_ami__model_A \
  higgs_evals/open_asr_ami__model_B \
  --output-csv data/ami_modelB_regressions.csv
```

*Note: You can skip this step if you just want to analyze overall errors across datasets using `convert_all_evals.py`.*

### 2. Categorize Errors via Pipeline

Run ErrorMap on the collected datasets/regressions against the taxonomy:

```bash
python run_errormap.py \
  --dataset ami_modelB_regressions \
  --model model_B \
  --asr \
  --reuse-taxonomy taxonomies/asr_base_v2_template.json \
  --exp-id comparison_modelA_modelB
```

### 3. Generate Aggregated CSV/Markdown Report

Finally, generate the quantitative aggregation and qualitative report to analyze error distributions without manually reading individual rows:

```bash
python report.py output/exp_name=construct_taxonomy_recursively__exp_id=comparison_modelA_modelB.csv --examples 2
```

This will print markdown tables showing:
- WER by Benchmark (corpus WER + critical error % + length breakdown, split by FLEURS / Open ASR)
- Overall Error Distribution (across models)
- Per-Benchmark Top 3 Error Categories
- Representative examples per category (with target and prediction)

`report.py` auto-detects the exp-id from the filename and enriches examples with `correct_answer` from the data_preparation cache. It also loads full benchmark data from `data/` for WER metrics — no re-run needed.

**Definitions used in the report:**
- Short: ≤ 5 reference words; Medium: 6–50; Long: > 50
- Critical Error %: share of utterances with WER ≥ 15% (score ≤ 0.85) — the population entering ErrorMap

---

## Instructions for AI Assistants — Reporting & Analysis Code

When writing or modifying reporting scripts (`report.py` or similar), follow these rules:

1. **Respect SRP in data functions.** Load functions only load. Enrichment, joins, and transformations belong in separate functions.

2. **Derive from context, don't add flags.** If a value is already encoded in an existing input (e.g., exp-id in the filename), parse it — don't add a redundant CLI argument.

3. **WER and quality metrics must always use the full benchmark population.** The ErrorMap output CSV contains only sampled/capped errors — never compute WER or Critical Error % from it. Always load from `data/` for any population-level metric.

4. **Never aggregate length-bucket metrics across benchmarks.** Short/medium/long WER breakdown is only meaningful within a single benchmark. Pooling across datasets with different characteristics is misleading.

5. **Combine related metrics into one table.** If two tables share the same grouping key (e.g., per-benchmark), they belong in one table, not two.

6. **Prefer one scannable master table over many small sub-tables.** N per-benchmark sub-tables require vertical scrolling and make cross-dataset comparison impossible. Use one flat table per logical group (e.g., FLEURS, Open ASR).
