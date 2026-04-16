#!/bin/bash
# Copy the latest .eval file for a specific (model, tag) pair into a namespaced destination.
#
# Usage:
#   bash copy_latest_evals.sh <model> <tag> [dest_dir]
#
# Examples:
#   bash copy_latest_evals.sh higgs-audio-m3 erik-v3
#   bash copy_latest_evals.sh higgs-audio-m3 chk65k-v3 ./higgs_evals/higgs-audio-m3__chk65k-v3

MODEL=${1:?"Usage: $0 <model> <tag> [dest_dir]"}
TAG=${2:?"Usage: $0 <model> <tag> [dest_dir]"}
DEST_DIR=${3:-"./higgs_evals/${MODEL}__${TAG}"}

mkdir -p "$DEST_DIR"

BASE_DIR="/hot-data/project/evaluation/results/inspectai-api/${MODEL}"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "Error: base directory not found: $BASE_DIR" >&2
    exit 1
fi

echo "Model : $MODEL"
echo "Tag   : $TAG"
echo "Dest  : $DEST_DIR"
echo ""

found=0

for bench_dir in "$BASE_DIR"/*/; do
    [ -e "$bench_dir" ] || continue

    bench=$(basename "$bench_dir")

    if [[ "$bench" =~ ^open_asr_.*$ ]] || [[ "$bench" =~ ^fleurs_asr_[^_]+$ ]]; then
        tag_dir="$bench_dir$TAG"

        if [[ -d "$tag_dir" ]]; then
            selected=""
            while IFS= read -r candidate; do
                if python3 -c "
import zipfile, sys
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        z.open('summaries.json').read(1)
    sys.exit(0)
except Exception as e:
    print(f'  [invalid] {sys.argv[1]}: {e}', flush=True)
    sys.exit(1)
" "$candidate" 2>/dev/null; then
                    selected="$candidate"
                    break
                else
                    echo "  $bench → $(basename "$candidate") [CORRUPT, skipping]"
                fi
            done < <(ls -1t "$tag_dir"/*.eval 2>/dev/null)

            if [[ -n "$selected" ]]; then
                filename=$(basename "$selected")
                dest_file="$DEST_DIR/${bench}__${TAG}_${filename}"
                echo "  $bench → $filename"
                cp "$selected" "$dest_file"
                ((found++))
            else
                echo "  $bench → no valid .eval found (all corrupt or missing)"
            fi
        fi
    fi
done

if [[ $found -eq 0 ]]; then
    echo "Warning: no .eval files found for tag '$TAG' under $BASE_DIR" >&2
    exit 1
fi

echo ""
echo "Done — copied $found file(s) to $DEST_DIR"
