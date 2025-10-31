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

label_path=/pl/active/krishnanlab/projects/metahq/labels/sampleLASSO-microarray__attribute-tissue__ontology-uberon__ecode-any.parquet
prob_path=/projects/$USER/txt2onto_prediction_workflow/results/predictions_prob_full
out_path=/projects/$USER/txt2onto_prediction_workflow/results

python threshold_annotation_analysis.py \
	-label_dir "$label_path" \
	-prob_dir "$prob_path" \
	-best_threshold "$out_path"/f0.5_best_threshold.csv \
	-annotations "$out_path"/sampleLASSO_microarry_sample_description_txt2onto2_pred_annotations_f05.parquet \
	-outdir "$out_path"/sampleLASSO_labels_prior_pos_predpos_f05.csv