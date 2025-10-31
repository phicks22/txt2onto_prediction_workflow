#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --partition=amilan
#SBATCH --qos=normal

module load miniforge
conda activate text-omics

cd /projects/$USER/txt2onto_prediction_workflow/src
data_dir=/scratch/alpine/$USER/2025-07-03_txt2onto2.0_workflow/data


python preprocess.py \
    -input "$data_dir"/corpus__level-series__organism-all__tech-all__stem-False__lemmatize-False.tsv \
    -out_text "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_descriptions.txt \
    -out_ids "$data_dir"/corpus__level-series__organism-all__tech-all__stem-True__lemmatize-True_ids.txt