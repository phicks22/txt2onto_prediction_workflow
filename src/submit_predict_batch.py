"""
This script performs prediction on the data in each chunk and submits the jobs by batch.

Parameters:
    -mem: it uses about 85GB memory to make predictions on 26,000 examples (260,000/10chunks). 
        For the sake of safty, request 96GB. 
    -time: it takes 40-50 mins to make predictions on 26,000 examples for one term. 
        As there are 15 terms, by default, to run within each job, it takes about 13hrs
        to complete a job. So, it is safe to request about 15hrs. 
Note: in the slurm progress printout, there are time and memory trace showing. Please check
there for the actual time and memory use. 

Authors: Junxia Lin
Date: 2025-10-07
"""

from argparse import ArgumentParser
from utils import check_outdir
import subprocess
import os
from pathlib import Path
import numpy as np

HOME_DIR = Path(__file__).resolve().parents[1]
SLURM_DIR = HOME_DIR / "slurms"
SRC_DIR = HOME_DIR / "src"
RUN_DIR = HOME_DIR / "run/submit"

def check(a):
    if len(a) == 0:
        print("Done!")
        return True
    else:
        print(f"There are {len(a)} tasks to do")
        return False
        
def submit_command(command: str) -> None:
    try:
        # Run the command in a shell
        result = subprocess.run(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print("Output:", result.stdout)
        else:
            print("Error:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        
def submit_predict_batch(
    input:str,
    out: str,
    input_embed: str,
    train_embed: str,
    model_path: str,
    batch_size: int,
    chunk_id: str,
    nodes: int,
    ntasks: int,
    mem: str,
    time: str, 
    partition: str,
    qos: str,
    ) -> None:

    # number of MONDO models
    model_paths = sorted(Path(model_path).glob("MONDO_*.pkl"))
    mondo_ids = [p.stem.split('__')[0] for p in model_paths]
    
    # check the finished ones
    finished_path = sorted(Path(out).glob("MONDO_*.csv"))
    finished_ids = [p.stem.split('__')[0] for p in finished_path]
    
    # Get unfinished IDs and path
    unfinished_ids = [mid for mid in mondo_ids if mid not in finished_ids]
    unfinished_paths = [p for p in model_paths if p.stem.split('__')[0] in unfinished_ids]

    if check(unfinished_ids):
        return
    count = len(unfinished_paths)
    n_split = max(1, count // batch_size)
    
    for y, models in enumerate(np.array_split(unfinished_paths, n_split)):
        job_script = []
        job_script.append("#!/bin/bash")
        job_script.append(f"#SBATCH --nodes={nodes}")
        job_script.append(f"#SBATCH --ntasks={ntasks}")
        job_script.append(f"#SBATCH --mem={mem}")
        job_script.append(f"#SBATCH --time={time}")
        job_script.append(f"#SBATCH --partition={partition}")
        job_script.append(f"#SBATCH --qos={qos}")
        job_script.append(f"#SBATCH --output={SLURM_DIR}/progress_%j.out")
        job_script.append("module load miniforge")
        job_script.append("conda activate txt2onto2")
        job_script.append(f"cd {SRC_DIR}")

        # Create command
        for model in models:
            cmd = f"python predict.py \
                    -input {input}/docs_part_{chunk_id}.txt \
                    -id {input}/ids_part_{chunk_id}.txt \
                    -out {out} \
                    -input_embed {input_embed} \
                    -train_embed {train_embed} \
                    -model {model}"
            
            job_script.append(cmd)

        # Write submission file and run
        submission_dir = check_outdir(RUN_DIR / f"prediction_batch")
        sb_file = submission_dir / f"txt2onto_4prediction_chunk{chunk_id}_batch{y}.sh"
        with open(sb_file, "w") as f:
            for line in job_script:
                f.write(f"{line}\n")

        submit_command(f"sbatch {sb_file}")
    print(f"{n_split} jobs were submited for chunk {chunk_id}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-input",
                        help="input path for description and id",
                        required=True,
                        type=str)
    parser.add_argument("-out",
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
    parser.add_argument("-chunk_id",
                        help="The nth chunk to run",
                        type=int,
                        required=True)
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
    
    submit_predict_batch(
        input=args.input,
        out=args.out,
        input_embed=args.input_embed,
        train_embed=args.train_embed,
        model_path=args.model_path,
        batch_size=args.batch_size,
        chunk_id=args.chunk_id,
        nodes=args.nodes,
        ntasks=args.ntasks,
        time=args.time,
        partition=args.partition,
        qos=args.qos,
        mem=args.mem,
    )