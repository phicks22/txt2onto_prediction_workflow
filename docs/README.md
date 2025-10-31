# Workflow for Generating Predicted Annotations on Study Descritpions Related to Specific Diseases

This repository contains a  workflow for generating predicted annotations on study descritpions related to specific diseases

## Directory Structure

- **bin**:
   - Contains MONDO model files (`*.pkl`) that are used to make predictions. These models are pretrained and ready to use with the provided data.
-  **data**: 
     - `true_label__inst_type=study__task=disease.csv.gz`: Compressed CSV file with true labels that includes redundant and non-redundant MONDO terms.
- **src**: 
  - `extract_data.py`: Script to extract descriptions and accession codes from the compressed JSON metadata file.
  - `preprocess.py`: Script to  preprocess the extracted descriptions.
  - `embedding_lookup_table.py`: Script to generate embeddings for preprocessed descriptions.
  - `tfidf_calculator.py`: Script to calculate TF-IDF scores for text data.
  - `predict.py`: Script to run predictions using pre-trained MONDO models.
  - `predict_chunk_batch.py`: Script to run predictions by the chunk of data and batch of diseases using pre-trained MONDO models. 
  - `split_descriptions_ids_into_chunks.py`: Script to split the descriptions and ids data into chunks. 
  - `submit_predict_batch.py`: Script to submit the predictions by batch. 
  - `combine_chunks_prediction.py`: Script to combine the predictions from each chunk of the data. 
  - `fbeta_binary_classification.py`: Script to generate the annotations by applying fbeta method to find the best threshold. 
  - `threshold_annotation_analysis.py`: Script to generate the statistics used for threshold & annotation analysis. 

- **results**: Contains the predictions, resulting best threshold statistics, and predicted annotations. 

- **run**: 
  - `run_extraction.sh`: Shell script for extracting and filtering descriptions.
  - `preprocess_study.sh`: Shell script to preprocess the extracted descriptions.
  - `embeddings_study.sh`: Shell script to generate embeddings for preprocessed descriptions.
  - `predict_chunk_batch.sh`: Shell script to run predictions using the MONDO model files by chunk and batch.
  - `combine_chunks_prediction.sh`: Shell script to combine the resutling predictions of each chunk.
  - `fbeta_classification_study.sh`: Shell script to classify the prediction probabilities into binary class and generate annotations. 
  - `threshold_annotation_analysis_study.sh`: Shell script to generate necessary threshold and annotation statistics. 

- **README.md**: This file, providing an overview of the workflow.

## Workflow Overview

### 1. Data Extraction and Filtering
- **Extract Descriptions**: The script `extract_data.py` reads and parses the compressed JSON metadata file located in `data/aggregated_metadata.json.gz`. It filters out entries with no descriptions.
  - Output: Filtered descriptions saved in `results/refinebio_descriptions_filtered.tsv`.
  - Accession codes saved in `results/IDs.tsv`.

### 2. Preprocess the Extracted Descriptions
- **Text Preprocessing**: The `preprocess_study.py` script cleans and preprocesses the extracted descriptions by removing URLs, specific strings, file names, non-UTF-8 characters, and applying text normalization techniques.

### 3. Generate Embeddings for Processed Descriptions
- **Embedding Generation**: The `embeddings_study.sh` script calls `embedding_lookup_table.py` to generate embeddings for the preprocessed descriptions using a pre-trained language model (BiomedBERT).

### 4. Run Predictions Using MONDO Model Files
- **Predictions**: The `predict_chunk_batch.py` script is used to run predictions for each MONDO model file using the generated embeddings and preprocessed descriptions. The `predict_chunk_batch.py` script calls the necessary functions from `split_descriptions_ids_into_chunks.py` and `submit_predict_batch.py` to first split the data into chunks and then make predictions for the data in each chunk. 
- **Combining Predictions**: Before combining the predcitions, please check the prediction results and make sure all the terms' prediction are completed across all the chunks. If any of the chunks have missing terms' prediction, re-run the `predict_chunk_batch.py` script to complete the remaining ones. After completing all the predictions, run the `combine_chunks_prediction.py` script to combine all the chunks' precition together for each disease. 
  - Output: Prediction results saved in `results` folder. This script needs also this `disease_desc_embedding.npz` to run.
  - Note: 
    - Adjust the `n_chunks` to change the number of data records in each chunk. The increase of this parameter will reduce the number of data in each chunk, which will faster the computational time for the prediction in each chunk. However, it will increase the number of slurm jobs needed to submit. 
    - Adjust the `batch_size` to change the number of disease terms in each slurm job. The increase of this parameter will increase the number of disease terms to run in each job, which will make the job take longer time. However, it will reduce the number of slurm jobs needed to submit.  
    - The relationship between this two parameters is: `total_slurm_job = total_term/batch_size * n_chunks`. Please note that the maximum number of slurm jobs allowed to submit to alpine at the same time is 1000. 
    - The `predict_chunk_batch.py` script will automatically scan the completed terms in each chunk in the directory and skip them. You can re-run the script to finish the predictions for the remaining terms if there are any left. 

### 5. Run fbeta binary classification
- **fbeta binary classification**: The `fbeta_binary_classification.py` apply fbeta method to classify predictions into binary classes and generate the annotations for each disease term. 

### 6. Generate threshold and annotation statistics
- **statistics**: The `threshold_annotation_analysis.py` generates the necessary statistics used to analyze the generated best thresholds and annotations.

