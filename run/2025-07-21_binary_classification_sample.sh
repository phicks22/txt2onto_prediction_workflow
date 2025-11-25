#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=05:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal

module load miniforge
conda activate text-omics
cd /projects/$USER/txt2onto_prediction_workflow/src

label_path=/pl/active/krishnanlab/projects/ml_gene_signatures/labels/metahq
pred_path=/projects/$USER/txt2onto_prediction_workflow/results/sampleLasso
sys_desc_path=/projects/$USER/txt2onto_prediction_workflow/data
method=fbeta
beta=1.0

python binary_classification.py \
	--label_dir "$label_path"/attribute-tissue__level-sample__mode-label__ecode-expert__species-human__tech-microarray.parquet \
	--prob_dir "$pred_path"/predictions_prob \
	--index_out_dir "$pred_path" \
	--outdir "$pred_path"/sampleLASSO_microarry_sample_description_txt2onto2_pred_annotations_f1.parquet \
	--constraints "$sys_desc_path"/uberon_system_descendants.json \
	--method "$method" \
	--beta "$beta" \
	--index_col sample \
	--group_col series
