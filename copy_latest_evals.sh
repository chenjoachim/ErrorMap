#!/bin/bash

# Target folder where the latest .eval files will be copied
# You can pass the target folder as the first argument, defaults to "latest_evals"
DEST_DIR=${1:-"./latest_evals"}
mkdir -p "$DEST_DIR"

BASE_DIR="/hot-data/project/evaluation/results/inspectai-api/higgs-audio-m3"

# The specific tags to look for
TAGS=(
    "chk65k-v3"
    "erik-v3"
    "20260309-erik-base"
    "20260319-erik-v3"
)

echo "Copying latest .eval files to: $DEST_DIR"

# Loop through all available benchmarks
for bench_dir in "$BASE_DIR"/*/; do
    # Skip if BASE_DIR happens to be empty
    [ -e "$bench_dir" ] || continue
    
    bench=$(basename "$bench_dir")
    
    # Check if benchmark is open_asr_* (can contain extra underscores)
    # or fleurs_asr_* (with no additional underscores)
    if [[ "$bench" =~ ^open_asr_.*$ ]] || [[ "$bench" =~ ^fleurs_asr_[^_]+$ ]]; then
        
        for tag in "${TAGS[@]}"; do
            tag_dir="$bench_dir$tag"
            
            # Check if the tag directory exists
            if [[ -d "$tag_dir" ]]; then
                
                # Fetch the latest .eval file sorted by modification time
                latest_eval=$(ls -1t "$tag_dir"/*.eval 2>/dev/null | head -n 1)
                
                if [[ -n "$latest_eval" ]]; then
                    filename=$(basename "$latest_eval")
                    echo "Found latest eval for $bench/$tag: $filename"
                    
                    # Optional: if you are worried about file name collisions, uncomment the below line
                    # and comment out the simple `cp` to prepend the benchmark and tag to the resulting file.
                    cp "$latest_eval" "$DEST_DIR/${bench}__${tag}_${filename}"

                fi
            fi
        done
    fi
done

echo "Done!"
