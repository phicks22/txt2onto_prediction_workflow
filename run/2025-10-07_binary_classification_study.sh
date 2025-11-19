#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=03:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal

module load miniforge
conda activate text-omics
cd /projects/$USER/txt2onto_prediction_workflow/src

label_path=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data/study_disease
pred_path=/projects/$USER/txt2onto_prediction_workflow/results/study
sys_desc_path=/projects/$USER/txt2onto_prediction_workflow/data
method=Balanced_accuracy

python binary_classification.py \
	--label_dir "$label_path"/true_label__inst_type=study__task=disease.parquet \
	--prob_dir "$pred_path"/predictions_prob/combined_files \
	--index_out_dir "$pred_path" \
	--outdir "$pred_path"/study_description_txt2onto2_pred_annotations_"$method".parquet \
	--constraints "$sys_desc_path"/mondo_system_descendants.json \
	--method "$method" \
	--is_study True