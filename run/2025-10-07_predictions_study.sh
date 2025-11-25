#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=print%j.out
#SBATCH --error=print%j.err
#SBATCH --ntasks=1
#SBATCH --mem=1024GB
#SBATCH --time=3:00:00
#SBATCH --partition=amem
#SBATCH --qos=mem


module load miniforge
conda activate txt2onto2
cd /projects/$USER/txt2onto2.0/src

data_dir=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data
out_dir=/projects/$USER/txt2onto_prediction_workflow/results/study

for model in ../bins/MONDO_*.pkl; do
  python predict.py \
  -input "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_descriptions.txt \
  -out "$out_dir"/predictions_prob/ \
  -id "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_ids.txt \
  -input_embed "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_descriptions_embedding.npz \
  -train_embed ../data/disease_desc_embedding.npz \
  -model "$model"
done