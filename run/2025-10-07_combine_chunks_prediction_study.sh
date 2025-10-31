#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal


module load miniforge
conda activate txt2onto2
cd /projects/$USER/txt2onto2.0/src

data_dir=/projects/$USER/txt2onto_prediction_workflow/results/study/predictions_prob
n_chunk=10

python combine_chunks_prediction.py \
    -input "$data_dir" \
    -n_chunk "$n_chunk" \
    -out "$data_dir"/combined_files