# Workflow for Public RNA-seq Studies Related to Specific Diseases

This repository contains a  workflow for finding public RNA-seq samples and studies related to specific diseases. 

## Directory Structure

- **bin**:
   - Contains MONDO model files (`*.pkl`) that are used to make predictions. These models are pretrained and ready to use with the provided data.
-  **data**: 
     - `aggregated_metadata.json.gz`: Compressed JSON file containing metadata about RNA-seq experiments from refine.bio
     - `true_label__inst_type=study__task=disease.csv.gz`: Compressed CSV file with true labels that includes redundant and non-redundant MONDO terms.
- **src**: 
  - `extract_data.py`: Script to extract descriptions and accession codes from the compressed JSON metadata file.
  - `preprocess.py`: Script to  preprocess the extracted descriptions.
  - `embedding_lookup_table.py`: Script to generate embeddings for preprocessed descriptions.
  - `tfidf_calculator.py`: Script to calculate TF-IDF scores for text data.
  - `predict.py`: Script to run predictions using pre-trained MONDO models.

- **results**: Contains the filtered descriptions and accession codes after preprocessing the metadata.
  - `IDs.tsv`: List of accession codes after filtering out studies with no description.
  - `refinebio_descriptions_filtered.tsv`: Descriptions of the RNA-seq experiments after filtering out studies with no description.
- **run**: 
  - `run_extraction.sh`: Shell script for extracting and filtering descriptions.
  - `run_embedding_lookup_table.sh`: Shell script to generate embeddings for preprocessed descriptions.
  - `run_preprocess.sh`: Shell script to preprocess the extracted descriptions.
  - `run_predictions.sh`: Shell script to run predictions using the MONDO model files.

- **README.md**: This file, providing an overview of the project.

## Workflow Overview

### 1. Data Extraction and Filtering
- **Extract Descriptions**: The script `extract_data.py` reads and parses the compressed JSON metadata file located in `data/aggregated_metadata.json.gz`. It filters out entries with no descriptions.
  - Output: Filtered descriptions saved in `results/refinebio_descriptions_filtered.tsv`.
  - Accession codes saved in `results/IDs.tsv`.

### 2. Preprocess the Extracted Descriptions
- **Text Preprocessing**: The `preprocess.py` script cleans and preprocesses the extracted descriptions by removing URLs, specific strings, file names, non-UTF-8 characters, and applying text normalization techniques.
  - Output: Preprocessed descriptions saved in `results/processed_refinebio_descriptions.tsv` for embedding generation.

### 3. Generate Embeddings for Processed Descriptions
- **Embedding Generation**: The `run_embedding_lookup_table.sh` script calls `embedding_lookup_table.py` to generate embeddings for the preprocessed descriptions using a pre-trained language model (BiomedBERT).
  - Output: Embeddings saved in `results/my_custom_embeddings.npz`.

### 4. Run Predictions Using MONDO Model Files
- **Predictions**: The `predict.py` script is used to run predictions for each MONDO model file using the generated embeddings and preprocessed descriptions.
  - Output: Prediction results saved in `prediction_results` folder. This script needs also this `txt2onto2.0/data/disease_desc_embedding.npz` to run.

## How to Run the Workflow

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/krishnanlab/Workflow_related_studies.git
   cd Workflow_related_studies
