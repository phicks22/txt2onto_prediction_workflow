#!/bin/bash

set -e  # Exit immediately if a command fails

echo "Step 1: Extract descriptions from metadata"
bash run/run_extraction.sh

echo "Step 2: Preprocess the filtered descriptions"
bash run/run_preprocess.sh

echo "Step 3: Generate embeddings"
bash run/run_embedding_lookup_table.sh

echo "Step 4: Run predictions using MONDO models"
bash run/run_predictions.sh

echo "Workflow completed successfully."
