# Dataset Documentation for Multimodal Sentiment Analysis

This directory contains the datasets used for the multimodal sentiment analysis project. The datasets include both raw and processed data, which are essential for training and evaluating the sentiment analysis model.

## Datasets

### Raw Data
The `raw` directory contains the original datasets used for sentiment analysis. The primary datasets included are:

- **CMU-MOSEI**: A large-scale dataset for multimodal sentiment analysis, containing video clips with corresponding text and audio annotations.
- **MuSe-CaR**: A dataset designed for multimodal sentiment analysis, featuring various emotional expressions across different modalities.

### Processed Data
The `processed` directory contains the datasets that have been pre-processed and are ready for model training and evaluation. This includes:

- Tokenized text data.
- Extracted audio features.
- Processed video frames and facial expression data.

## Accessing the Datasets
To access the datasets, please follow these steps:

1. **Download the Raw Datasets**: Obtain the datasets from their respective sources and place them in the `data/raw` directory.
2. **Process the Data**: Use the provided scripts in the `src` directory to process the raw data and save the processed datasets in the `data/processed` directory.

## Usage
The processed datasets can be directly used for training and evaluating the multimodal sentiment analysis model. Ensure that the data is correctly formatted and accessible to the model training scripts.

For further details on how to process the data and utilize it in the model, refer to the documentation in the `src` directory and the Jupyter notebooks in the `notebooks` directory.