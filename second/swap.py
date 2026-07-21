#!/usr/bin/env python3

import pandas as pd
import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} input.csv output.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Read CSV
df = pd.read_csv(input_file)

# Reorder columns
df = df[["velocity", "sin", "cos", "epoch"]]

# Save
df.to_csv(output_file, index=False)

print(f"Saved reordered CSV to {output_file}")
