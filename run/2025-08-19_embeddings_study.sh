#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu
#SBATCH --qos=normal

module load miniforge
conda activate text-omics
# run the following two lines at the first time, or use txt2onto2 environment
#pip install torch
#pip install transformers
cd /projects/$USER/txt2onto_prediction_workflow/src

data_dir=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data
python embedding_lookup_table.py \
    -input "$data_dir"/corpus__level-series__organism-all__tech-microarray_rnaseq__stem-True__lemmatize-True_descriptions.txt \
    -out "$data_dir"/corpus__level-series__organism-all__tech-microarray_rnaseq__stem-True__lemmatize-True_descriptions_embedding.npz \
    -model biomedbert_abs \
    -batch_size 2000