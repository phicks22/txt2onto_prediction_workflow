#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal

module load miniforge
conda activate text-omics
cd /projects/$USER/txt2onto_prediction_workflow/src

pred_path=/projects/$USER/txt2onto_prediction_workflow/results/study/predictions_prob

python combine_predictions_all_terms.py \
    -in_dir "$pred_path"/combined_files \
    -out_csv "$pred_path"/combine_term_preds_study_100.csv