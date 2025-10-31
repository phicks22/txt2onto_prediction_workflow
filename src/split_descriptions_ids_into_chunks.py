"""
This script split the descriptions and ids into chunks.

Authors: Junxia Lin
Date: 2025-10-07
"""

from argparse import ArgumentParser
import os

def split_desc_id_chunk(
    doc_path: str,
    idx_path: str,
    n_chunks: int,
    outdir: str,
    ) -> None:
    
    # Read the ids and descriptions from the files, skipping the header
    with open(idx_path, 'r') as f_ids, open(doc_path, 'r') as f_desc:
        ids = f_ids.readlines()[1:]
        descs = f_desc.readlines()[1:]
    
    # Calculate the size of each chunk
    chunk_size = len(ids) // n_chunks
    
    os.makedirs(outdir, exist_ok=True)
    # Split the data into chunks and write to new files
    for i in range(n_chunks):
        start_index = i * chunk_size
        # Ensure the last chunk includes any remaining items
        end_index = (i + 1) * chunk_size if i < n_chunks - 1 else len(ids)
    
        # Write ids chunk
        with open(f'{outdir}/ids_part_{i}.txt', 'w') as f_ids_chunk:
            f_ids_chunk.writelines(ids[start_index:end_index])
    
        # Write descriptions chunk
        with open(f'{outdir}/docs_part_{i}.txt', 'w') as f_desc_chunk:
            f_desc_chunk.writelines(descs[start_index:end_index])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-input",
                        help="description",
                        required=True,
                        type=str)
    parser.add_argument("-id",
                        help="input instance id",
                        required=True,
                        type=str)
    parser.add_argument("-out",
                        help="outdir",
                        required=True,
                        type=str)
    parser.add_argument("-n_chunks",
                        help="Number of chunks to split the data into",
                        type=int,
                        default=10)
    args = parser.parse_args()
    
    split_desc_id_chunk(idx_path=args.id,
                        doc_path=args.input,
                        outdir=args.out,
                        n_chunks=args.n_chunks
                        )
