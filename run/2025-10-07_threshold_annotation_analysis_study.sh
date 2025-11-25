#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal

module load miniforge
conda activate text-omics
cd /projects/$USER/txt2onto_prediction_workflow/src

label_path=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data/study_disease
prob_path=/projects/$USER/txt2onto_prediction_workflow/results/study

python threshold_annotation_analysis.py \
	-label_dir "$label_path"/true_label__inst_type=study__task=disease.parquet \
	-prob_dir "$prob_path"/predictions_prob_old/combined_files \
	-best_threshold "$prob_path"/Balanced_accuracy_best_threshold.csv \
	-annotations "$prob_path"/study_description_txt2onto2_pred_annotations_Balanced_accuracy.parquet \
	-outdir "$prob_path"/threshold_annotation_analysis_Balanced_accuracy.csv \
	-is_study True