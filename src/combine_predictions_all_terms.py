"""
This script combines all the predictions of the terms into a single file. 

Authors: Junxia Lin
Date: 2025-11-06
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import re

def mondo_term_from_name(p: Path) -> str:
    m = rx.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse MONDO id from filename: {p.name}")
    return f"MONDO:{m.group(1)}"
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-in_dir",
        help="Path to the predictions", 
        required=True, 
        type=str
    )
    parser.add_argument(
        "-out_csv",
        help="Output path",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    
    
    rx = re.compile(r"MONDO_(\d+)__preds") 
        
    dfs = []
    for fp in sorted(Path(args.in_dir).glob("MONDO_*__preds.csv"))[:100]:
        term = mondo_term_from_name(fp)
        df = pd.read_csv(fp)
        df.insert(0, "term", term)
        dfs.append(df)
    
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out):,} rows to {args.out_csv}")