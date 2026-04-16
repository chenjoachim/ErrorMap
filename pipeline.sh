#!/bin/bash
# End-to-end ErrorMap pipeline: model checkpoint → error report.
#
# Usage:
#   bash pipeline.sh --model <model> --tag <tag> [options]
#
# Examples:
#   bash pipeline.sh --model higgs-audio-m3 --tag erik-v3
#   bash pipeline.sh --model higgs-audio-m3 --tag chk65k-v3 --max-per-dataset 200
#
# The megatag "<model>__<tag>" is used to namespace all intermediate and output
# directories so runs never overwrite each other.

set -euo pipefail

# ---------------------------------------------------------------------------
# Predefined benchmark list — edit to add/remove benchmarks.
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARKS=(
    open_asr_ami
    open_asr_earnings22
    open_asr_gigaspeech
    open_asr_librispeech_clean
    open_asr_librispeech_other
    open_asr_spgispeech
    open_asr_tedlium
    open_asr_voxpopuli
    fleurs_asr_en
    fleurs_asr_es
    fleurs_asr_ja
    fleurs_asr_ko
    fleurs_asr_zh
)

# ---------------------------------------------------------------------------
# Defaults for ErrorMap pipeline knobs
# ---------------------------------------------------------------------------
THRESHOLD=0.85
MAX_PER_DATASET=50
TAXONOMY="taxonomies/asr_base_v2_template.json"
EXAMPLES=1

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MODEL=""
TAG=""
BENCHMARKS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";           shift 2 ;;
        --tag)           TAG="$2";             shift 2 ;;
        --benchmarks)    shift; while [[ $# -gt 0 && "$1" != --* ]]; do BENCHMARKS+=("$1"); shift; done ;;
        --threshold)     THRESHOLD="$2";       shift 2 ;;
        --max-per-dataset) MAX_PER_DATASET="$2"; shift 2 ;;
        --taxonomy)      TAXONOMY="$2";        shift 2 ;;
        --examples)      EXAMPLES="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$TAG" ]]; then
    echo "Usage: bash pipeline.sh --model <model> --tag <tag> [options]" >&2
    exit 1
fi

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
    BENCHMARKS=("${DEFAULT_BENCHMARKS[@]}")
fi

# ---------------------------------------------------------------------------
# Derived paths — all namespaced under the megatag
# ---------------------------------------------------------------------------
MEGATAG="${MODEL}__${TAG}"
EVALS_DIR="higgs_evals/${MEGATAG}"
DATA_DIR="data/${MEGATAG}"
EXP_ID="${MEGATAG}"
OUTPUT_CSV="output/exp_name=construct_taxonomy_recursively__exp_id=${EXP_ID}.csv"
REPORT_FILE="reports/${MEGATAG}.md"

echo "============================================================"
echo "  ErrorMap Pipeline"
echo "  Model   : $MODEL"
echo "  Tag     : $TAG"
echo "  Megatag : $MEGATAG"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Copy .eval files
# ---------------------------------------------------------------------------
echo "── Step 1: Copy .eval files → $EVALS_DIR"
bash copy_latest_evals.sh "$MODEL" "$TAG" "$EVALS_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Convert .eval → CSV
# ---------------------------------------------------------------------------
echo "── Step 2: Convert .eval files → $DATA_DIR/"
python convert_all_evals.py --evals-dir "$EVALS_DIR" --output-dir "$DATA_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run ErrorMap pipeline
# ---------------------------------------------------------------------------
echo "── Step 3: Run ErrorMap pipeline (exp-id: $EXP_ID)"
python run_errormap.py \
    --dataset "${BENCHMARKS[@]}" \
    --model "$TAG" \
    --asr \
    --reuse-taxonomy "$TAXONOMY" \
    --threshold "$THRESHOLD" \
    --max-per-dataset "$MAX_PER_DATASET" \
    --data-path "$DATA_DIR" \
    --exp-id "$EXP_ID"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Generate report
# ---------------------------------------------------------------------------
echo "── Step 4: Generate report → $REPORT_FILE"
if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "Error: expected output not found: $OUTPUT_CSV" >&2
    exit 1
fi
mkdir -p reports
python report.py "$OUTPUT_CSV" \
    --examples "$EXAMPLES" \
    --category-col category_depth_1 \
    --data-dir "$DATA_DIR" \
    | tee "$REPORT_FILE"
