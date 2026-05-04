#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate rows and columns of a CSV file")
    parser.add_argument("csv_path", nargs="?", default="winequality-white.csv",
                        help="Path to CSV file (default: winequality-white.csv)")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(csv_path)
    rows, cols = df.shape
    print(f"Rows: {rows}")
    print(f"Columns: {cols}")


if __name__ == "__main__":
    main()
