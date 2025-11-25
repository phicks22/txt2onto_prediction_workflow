#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=print%j.out
#SBATCH --error=print%j.err
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal


module load miniforge
conda activate txt2onto2
cd /projects/$USER/txt2onto2.0/src

data_dir=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data
out_dir=/projects/$USER/txt2onto_prediction_workflow/results
for model in ../bins/{CL_,UBERON_}*.pkl; do
  python predict.py \
  -input "$data_dir"/sampleLASSO-microarray_txt2onto_preprocessed_sample_descriptions_full.txt \
  -out "$out_dir"/predictions_prob_full/ \
  -id "$data_dir"/sampleLASSO-microarray_txt2onto_preprocessed_sample_ids_full.txt \
  -input_embed "$data_dir"/sampleLASSO-microarray_txt2onto_preprocessed_descriptions_embedding_full.npz \
  -train_embed ../data/tissue_desc_embedding.npz \
  -model "$model"
done