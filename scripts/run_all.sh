#!/bin/bash
# Run paper_filter for all supported conferences and years.
# Usage:
#   ./scripts/run_all.sh download                                    # Download all PDFs
#   ./scripts/run_all.sh process "topic" [threshold] [description]   # Process all
#   ./scripts/run_all.sh all "topic" [threshold] [description]       # Download + Process
#
# Examples:
#   ./scripts/run_all.sh download
#   ./scripts/run_all.sh process "agentic AI systems" 8
#   ./scripts/run_all.sh all "test-time learning" 8 "Methods that adapt models during inference"

set -e

COMMAND="${1:?Usage: $0 <download|process|all> [topic] [threshold] [description]}"
shift

OUTPUT_DIR="output"
LOG_LEVEL="INFO"

declare -a JOBS=(
    "cvpr 2021"
    "cvpr 2022"
    "cvpr 2023"
    "cvpr 2024"
    "cvpr 2025"
    "iccv 2021"
    "iccv 2023"
    "iccv 2025"
    "eccv 2022"
    "eccv 2024"
    "iclr 2024"
    "iclr 2025"
    "iclr 2026"
    "icml 2021"
    "icml 2022"
    "icml 2023"
    "icml 2024"
    # "icml 2025"       # blocked
    "neurips 2021"
    "neurips 2022"
    "neurips 2023"
    "neurips 2024"
    "neurips 2025"
    "emnlp 2021"
    "emnlp 2022"
    "emnlp 2023"
    "emnlp 2024"
    "emnlp 2025"
    "aaai 2021"
    "aaai 2022"
    "aaai 2023"
    "aaai 2024"
    "aaai 2025"
    "aaai 2026"
)

run_download() {
    echo "============================================================"
    echo "Paper Filter — Download All Conferences"
    echo "============================================================"

    local TOTAL=${#JOBS[@]} PASSED=0 FAILED=0
    local FAILED_LIST=()

    for i in "${!JOBS[@]}"; do
        local CONF=$(echo "${JOBS[$i]}" | awk '{print $1}')
        local YEAR=$(echo "${JOBS[$i]}" | awk '{print $2}')
        local CONF_UPPER=$(echo "$CONF" | tr '[:lower:]' '[:upper:]')

        echo ""
        echo "[$((i+1))/$TOTAL] download $CONF $YEAR"

        if python -m paper_filter download -c "$CONF" -y "$YEAR" \
            --output-dir "$OUTPUT_DIR" --log-level "$LOG_LEVEL"; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$CONF $YEAR")
        fi
    done

    echo ""
    echo "Download complete: $PASSED OK, $FAILED failed"
    [ ${#FAILED_LIST[@]} -gt 0 ] && echo "Failed: ${FAILED_LIST[*]}"
}

run_process() {
    local TOPIC="${1:?process requires a topic}"
    local THRESHOLD="${2:-8}"
    local TOPIC_DESC="${3:-}"

    echo "============================================================"
    echo "Paper Filter — Process All Conferences"
    echo "Topic:     $TOPIC"
    [ -n "$TOPIC_DESC" ] && echo "Scope:     $TOPIC_DESC"
    echo "Threshold: $THRESHOLD"
    echo "============================================================"

    local TOTAL=${#JOBS[@]} PASSED=0 FAILED=0
    local FAILED_LIST=()

    for i in "${!JOBS[@]}"; do
        local CONF=$(echo "${JOBS[$i]}" | awk '{print $1}')
        local YEAR=$(echo "${JOBS[$i]}" | awk '{print $2}')
        local CONF_UPPER=$(echo "$CONF" | tr '[:lower:]' '[:upper:]')
        local PDF_DIR="$OUTPUT_DIR/pdfs/${CONF_UPPER}_${YEAR}"

        echo ""
        echo "[$((i+1))/$TOTAL] process $CONF $YEAR"

        # Skip if no PDF dir exists
        if [ ! -d "$PDF_DIR" ]; then
            echo "  No PDFs at $PDF_DIR — skipping"
            continue
        fi

        local DESC_ARG=""
        [ -n "$TOPIC_DESC" ] && DESC_ARG="-d \"$TOPIC_DESC\""

        if eval python -m paper_filter process \
            --pdf-dir "\"$PDF_DIR\"" \
            -c "$CONF" -y "$YEAR" \
            -t "\"$TOPIC\"" \
            $DESC_ARG \
            --relevance-threshold "$THRESHOLD" \
            --output-dir "$OUTPUT_DIR" \
            --log-level "$LOG_LEVEL"; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$CONF $YEAR")
        fi
    done

    echo ""
    echo "Process complete: $PASSED OK, $FAILED failed"
    [ ${#FAILED_LIST[@]} -gt 0 ] && echo "Failed: ${FAILED_LIST[*]}"
}

case "$COMMAND" in
    download) run_download ;;
    process)  run_process "$@" ;;
    all)
        run_download
        echo ""
        run_process "$@"
        ;;
    *) echo "Usage: $0 <download|process|all> [topic] [threshold] [description]"; exit 1 ;;
esac
