#!/bin/bash

# Define the common paths
INPUT="/Users/souhatifour/Downloads/results/processed_text.tsv"
OUT_DIR="/Users/souhatifour/Downloads/prediction_results"
ID_FILE="/Users/souhatifour/Downloads/results/retained_ids.tsv"
INPUT_EMBED="/Users/souhatifour/Downloads/results/my_custom_embeddings.npz"
TRAIN_EMBED="/Users/souhatifour/Downloads/disease_desc_embedding.npz"
MODEL_DIR="/Users/souhatifour/Downloads/Workflow_related_studies/bin/"

# Loop through each MONDO model file
for MODEL in ${MODEL_DIR}MONDO_*.pkl; do
    echo "Running prediction for model $MODEL"
    python /Users/souhatifour/Downloads/predict.py \
        -input $INPUT \
        -out $OUT_DIR \
        -id $ID_FILE \
        -input_embed $INPUT_EMBED \
        -train_embed $TRAIN_EMBED \
        -model $MODEL
done

