#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=amilan
#SBATCH --qos=normal

module load miniforge
conda activate txt2onto2
cd /projects/$USER/txt2onto2.0/src

data_dir=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data/study_disease
pred_dir=/projects/$USER/txt2onto_prediction_workflow/results/study
model_path=/projects/$USER/txt2onto2.0/bins
n_chunks=10
batch_size=15

python predict_chunk_batch.py \
    -desc "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_descriptions.txt \
    -id "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_ids.txt \
    -n_chunks "$n_chunks" \
    -chunk_dir "$data_dir"/description_id_chunks/ \
    -pred_dir "$pred_dir"/predictions_prob/ \
    -input_embed "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_descriptions_embedding.npz \
    -train_embed ../data/disease_desc_embedding.npz \
    -model_path "$model_path" \
    -batch_size "$batch_size" 