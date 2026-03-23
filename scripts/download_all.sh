#!/bin/bash
# Download all papers from all supported conferences and years.
# Usage: ./scripts/download_all.sh [output_dir]
#
# Example:
#   ./scripts/download_all.sh
#   ./scripts/download_all.sh /data/papers

set -e

OUTPUT_DIR="${1:-output}"
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
    "iclr 2021"
    "iclr 2022"
    "iclr 2023"
    "iclr 2024"
    "iclr 2025"
    "iclr 2026"
    "icml 2021"
    "icml 2022"
    "icml 2023"
    "icml 2024"
    "icml 2025"
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

TOTAL=${#JOBS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

echo "============================================================"
echo "Paper Filter — Download All Conferences"
echo "Output: $OUTPUT_DIR"
echo "Total:  $TOTAL conference-year combinations"
echo "============================================================"

for i in "${!JOBS[@]}"; do
    CONF=$(echo "${JOBS[$i]}" | awk '{print $1}')
    YEAR=$(echo "${JOBS[$i]}" | awk '{print $2}')

    echo ""
    echo "[$((i+1))/$TOTAL] $CONF $YEAR"

    if python -m paper_filter download \
        -c "$CONF" -y "$YEAR" \
        --output-dir "$OUTPUT_DIR" \
        --log-level "$LOG_LEVEL"; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$CONF $YEAR")
        echo "[FAIL] $CONF $YEAR"
    fi
done

echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED"

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo ""
    echo "Failed conferences:"
    for f in "${FAILED_LIST[@]}"; do
        echo "  - $f"
    done
fi

echo ""
echo "PDFs saved to: $OUTPUT_DIR/pdfs/"
echo "============================================================"
