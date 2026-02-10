#!/usr/bin/env python3
"""Helper script to parse QS World Rankings Excel file into CSV format.

Usage:
    python scripts/download_rankings.py --input qs_rankings.xlsx --output config/university_rankings.csv

The QS rankings Excel file can be downloaded from:
https://www.topuniversities.com/world-university-rankings

This script reads the Excel file and outputs a simple CSV with columns: rank, university
"""

import argparse
import csv
import sys

def parse_qs_excel(input_path: str, output_path: str, max_rank: int = 1000) -> None:
    """Parse QS rankings Excel file and write CSV."""
    try:
        import openpyxl
    except ImportError:
        print("Error: openpyxl is required. Install with: pip install openpyxl")
        sys.exit(1)

    wb = openpyxl.load_workbook(input_path, read_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        print("Error: Empty spreadsheet")
        sys.exit(1)

    # Try to find header row with 'rank' and 'institution'/'university'
    header_idx = None
    rank_col = None
    name_col = None

    for i, row in enumerate(rows):
        row_lower = [str(c).lower().strip() if c else "" for c in row]
        for j, cell in enumerate(row_lower):
            if "rank" in cell and rank_col is None:
                rank_col = j
                header_idx = i
            if any(kw in cell for kw in ["institution", "university", "name"]):
                name_col = j

        if rank_col is not None and name_col is not None:
            break

    if rank_col is None or name_col is None:
        print("Error: Could not identify rank and university columns.")
        print(f"First row: {rows[0]}")
        sys.exit(1)

    print(f"Found headers at row {header_idx}: rank_col={rank_col}, name_col={name_col}")

    entries = []
    for row in rows[header_idx + 1:]:
        rank_val = row[rank_col] if rank_col < len(row) else None
        name_val = row[name_col] if name_col < len(row) else None

        if rank_val is None or name_val is None:
            continue

        # Handle rank ranges like "101-150"
        rank_str = str(rank_val).strip()
        if "-" in rank_str:
            try:
                rank_num = int(rank_str.split("-")[0])
            except ValueError:
                continue
        elif rank_str.endswith("+"):
            try:
                rank_num = int(rank_str.rstrip("+"))
            except ValueError:
                continue
        else:
            try:
                rank_num = int(rank_str)
            except ValueError:
                continue

        if rank_num > max_rank:
            continue

        entries.append((rank_num, str(name_val).strip()))

    entries.sort(key=lambda x: x[0])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "university"])
        for rank, name in entries:
            writer.writerow([rank, name])

    print(f"Wrote {len(entries)} universities to {output_path}")
    wb.close()


def main():
    parser = argparse.ArgumentParser(description="Parse QS World Rankings Excel to CSV")
    parser.add_argument("--input", required=True, help="Path to QS rankings Excel file")
    parser.add_argument("--output", default="config/university_rankings.csv",
                        help="Output CSV path")
    parser.add_argument("--max-rank", type=int, default=1000,
                        help="Maximum rank to include")
    args = parser.parse_args()
    parse_qs_excel(args.input, args.output, args.max_rank)


if __name__ == "__main__":
    main()
