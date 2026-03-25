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

# Status key:
#   uncommented = needs downloading
#   # DONE      = fully downloaded
#   # BLOCKED   = requires manual download
#   # PARTIAL   = partially downloaded, re-run to resume

declare -a JOBS=(
    # DONE "cvpr 2021"       # 1660 PDFs
    # DONE "cvpr 2022"       # 2071 PDFs
    # DONE "cvpr 2023"       # 2351 PDFs
    # DONE "cvpr 2024"       # 2711 PDFs
    # DONE "cvpr 2025"       # 2870 PDFs
    # DONE "iccv 2021"       # 1612 PDFs
    # DONE "iccv 2023"       # 2155 PDFs
    # DONE "iccv 2025"       # 2700 PDFs
    # DONE "eccv 2022"       # 1645 PDFs
    # DONE "eccv 2024"       # 2379 PDFs
    # PARTIAL "iclr 2021"    # 45 PDFs (old fetcher, needs re-download)
    "iclr 2022"              # not yet downloaded (requires .env)
    "iclr 2023"              # not yet downloaded (requires .env)
    # DONE "iclr 2024"       # 2260 PDFs
    # DONE "iclr 2025"       # 3703 PDFs
    # DONE "iclr 2026"       # 5355 PDFs
    # DONE "icml 2021"       # 1183 PDFs
    # DONE "icml 2022"       # 1233 PDFs
    # DONE "icml 2023"       # 1828 PDFs
    # DONE "icml 2024"       # 2610 PDFs
    # DONE "icml 2025"       # 3305 PDFs (requires .env)
    # DONE "neurips 2021"    # 2334 PDFs
    # DONE "neurips 2022"    # 2834 PDFs
    # DONE "neurips 2023"    # 3540 PDFs
    # DONE "neurips 2024"    # 4169 PDFs (requires .env)
    # DONE "neurips 2025"    # 5347 PDFs (requires .env)
    # DONE "emnlp 2021"      # 889 PDFs
    # DONE "emnlp 2022"      # 893 PDFs
    # DONE "emnlp 2023"      # 1174 PDFs
    # DONE "emnlp 2024"      # 1436 PDFs
    # DONE "emnlp 2025"      # 1996 PDFs
    # DONE "aaai 2021"       # 1635 PDFs
    # DONE "aaai 2022"       # 1306 PDFs
    # DONE "aaai 2023"       # 1557 PDFs
    # DONE "aaai 2024"       # 2291 PDFs
    # DONE "aaai 2025"       # 3028 PDFs
    # DONE "aaai 2026"       # 4149 PDFs
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
