"""
This script:
1. splits the descriptions and ids into chunks, 
2. performs prediction on the data in each chunk and submits the jobs by batch.

Authors: Junxia Lin
Date: 2025-10-07
"""


from argparse import ArgumentParser
import os
from split_descriptions_ids_into_chunks import split_desc_id_chunk
from submit_predict_batch import submit_predict_batch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-desc",
                        help="input path to description",
                        required=True,
                        type=str)
    parser.add_argument("-id",
                        help="input path to id",
                        required=True,
                        type=str)
    parser.add_argument("-n_chunks",
                        help="Number of chunks to split the data into",
                        type=int,
                        default=10)
    parser.add_argument("-chunk_dir",
                        help="dir for the chunked data",
                        required=True,
                        type=str)
    parser.add_argument("-pred_dir",
                        help="outdir to prediction results",
                        required=True,
                        type=str)
    parser.add_argument("-input_embed",
                        help="word embedding for the input text",
                        required=True,
                        type=str)
    parser.add_argument("-train_embed",
                        help="word embedding for training text",
                        required=True,
                        type=str)
    parser.add_argument("-model_path",
                        help="Path for trained model",
                        required=True,
                        type=str)
    parser.add_argument("-batch_size",
                        help="number of terms (dieases/tissues) to run in each job",
                        type=int,
                        default=15)
    parser.add_argument("-nodes",
                        help="Number of nodes to request.",
                        type=int,
                        default=1)
    parser.add_argument("-ntasks",
                        help="How many processes to run on each node.",
                        type=int,
                        default=1)
    parser.add_argument("-time",
                        help="Time to run each submitted job.",
                        type=str,
                        default="15:30:00")
    parser.add_argument("-partition",
                        help="specify a partition in order for the job \
                              to run on the appropriate type of node.",
                        type=str,
                        default="amilan")
    parser.add_argument("-qos",
                        help="quality of service.",
                        type=str,
                        default="normal")
    parser.add_argument("-mem", 
                        help="Amount of memory to request.", 
                        type=str, 
                        default="96GB")
    args = parser.parse_args()
    
    # split the descriptions and ids into chunk
    split_desc_id_chunk(
        doc_path=args.desc,
        idx_path=args.id,
        n_chunks=args.n_chunks,
        outdir=args.chunk_dir,
        )
                        
    # performs prediction and submits the jobs by batch
    for c in range(args.n_chunks):
        
        pred_chunk_dir = f"{args.pred_dir}/chunk_{c}"
        os.makedirs(pred_chunk_dir, exist_ok=True)
        
        submit_predict_batch(
            input=args.chunk_dir,
            out=pred_chunk_dir,
            input_embed=args.input_embed,
            train_embed=args.train_embed,
            model_path=args.model_path,
            batch_size=args.batch_size,
            chunk_id=c,
            nodes=args.nodes,
            ntasks=args.ntasks,
            time=args.time,
            partition=args.partition,
            qos=args.qos,
            mem=args.mem,
            )

    