"""
This script combines the predictions from each chunk together for each term.

Authors: Junxia Lin
Date: 2025-10-07
"""

import polars as pl
from pathlib import Path
from tqdm import tqdm
from utils import check_outdir
from argparse import ArgumentParser

def combine_chunk(
    input:str, 
    n_chunk: int, 
    out: str,
    ) -> None:
                
    # List all batch directories
    input = Path(input)
    batch_dirs = [input / f'chunk_{i}' for i in range(n_chunk)]
    
    # Get the list of filenames from the first batch folder
    file_names = [f.name for f in batch_dirs[0].glob('*.csv')]
    
    # Create an output directory to store combined files
    out = check_outdir(out)
    
    # Combine CSV files with the same name across all batch folders
    for file_name in tqdm(file_names):
        dfs = []
        for batch_dir in batch_dirs:
            file_path = batch_dir / file_name
            if file_path.exists():
                df = pl.read_csv(file_path)
                dfs.append(df)
        
        # Concatenate all dataframes and save to a new CSV
        if dfs:
            combined_df = pl.concat(dfs, how="vertical")
            combined_df = combined_df.filter(pl.col("ID").str.contains("GSE|GSM"))
            combined_df = combined_df.unique(subset=["ID"], keep="first")
            combined_df.write_csv(out / file_name)
    
    print(f"Combined files saved to: {out}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-input",
                        help="input path for prediction chunk folders",
                        required=True,
                        type=str)
    parser.add_argument("-n_chunk",
                        help="Number of chunks that data is split into",
                        type=int,
                        default=10)
    parser.add_argument("-out",
                        help="output dir for the combined predictions",
                        required=True,
                        type=str)
    args = parser.parse_args()
    
    # number of chunks the data is split
    combine_chunk(
                input=args.input,
                n_chunk=args.n_chunk,
                out=args.out
                )

