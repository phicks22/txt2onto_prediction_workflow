#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --partition=amilan
#SBATCH --qos=normal


module load miniforge
conda activate txt2onto2
cd /projects/$USER/txt2onto2.0/src

data_dir=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data
out_dir=/projects/$USER/txt2onto_prediction_workflow/all_tech_study

python predict_batch.py \
    -input "$data_dir"/descriptions_ids_chunk \
    -out "$out_dir"/predictions_prob/ \
    -input_embed "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_descriptions_embedding.npz \
    -train_embed ../data/disease_desc_embedding.npz \
    -model ../bins/MONDO_0700066__model.pkl
