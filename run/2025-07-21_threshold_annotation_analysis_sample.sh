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

label_path=/pl/active/krishnanlab/projects/ml_gene_signatures/labels/metahq
prob_path=/projects/$USER/txt2onto_prediction_workflow/results/sampleLasso
method=Balanced_accuracy

python threshold_annotation_analysis.py \
	-label_dir "$label_path"/attribute-tissue__level-sample__mode-label__ecode-expert__species-human__tech-microarray.parquet \
	-prob_dir "$prob_path"/predictions_prob \
	-best_threshold "$prob_path"/"$method"_best_threshold.csv \
	-annotations "$prob_path"/sampleLASSO_microarry_sample_description_txt2onto2_pred_annotations_"$method".parquet \
	-outdir "$prob_path"/threshold_annotation_analysis_"$method".csv \
	-index_col sample \
	-group_col series