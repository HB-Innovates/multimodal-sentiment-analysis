# Multimodal Sentiment Analysis

This project aims to develop a system that integrates text, audio, and video data to predict sentiment using advanced machine learning techniques. The system leverages Transformer models to analyze multiple modalities and provide accurate sentiment predictions.

## Key Features

- **Multimodal Data Processing**: The project processes text, audio, and video data to extract meaningful features for sentiment analysis.
- **Transformer-Based Model**: Utilizes state-of-the-art Transformer architectures for effective sentiment prediction.
- **Dynamic Attention Mechanisms**: Implements attention mechanisms to weigh the importance of each modality during prediction.
- **Web Application**: Provides a user-friendly web interface for uploading data and obtaining sentiment predictions.

## Project Structure

- `data/`: Contains raw and processed datasets.
  - `raw/`: Raw datasets (e.g., CMU-MOSEI, MuSe-CaR).
  - `processed/`: Processed datasets ready for training and evaluation.
  - `README.md`: Documentation on datasets.

- `src/`: Source code for data processing, model training, and evaluation.
  - `text_processing.py`: Functions for text data processing using BERT or RoBERTa.
  - `audio_processing.py`: Functions for audio feature extraction using Librosa.
  - `video_processing.py`: Functions for video data processing using OpenCV.
  - `model.py`: Definition of the multimodal model architecture.
  - `train.py`: Training loop for the model.
  - `evaluate.py`: Model evaluation functions.
  - `utils.py`: Utility functions for data handling and visualization.

- `notebooks/`: Jupyter notebooks for data exploration, model training, and evaluation.
  - `data_exploration.ipynb`: Explore and visualize the dataset.
  - `model_training.ipynb`: Train the multimodal sentiment analysis model.
  - `model_evaluation.ipynb`: Evaluate the trained model and visualize results.

- `app/`: Web application files.
  - `main.py`: Entry point for the web application.
  - `templates/`: HTML templates for the web interface.
  - `static/`: Static files such as CSS styles.

- `requirements.txt`: Lists project dependencies.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd multimodal-sentiment-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the datasets by placing them in the `data/raw/` directory and processing them as needed.

4. Run the web application:
   ```
   python app/main.py
   ```

5. Access the application in your web browser at `http://localhost:5000`.

## Usage

Upload text, audio, and video files through the web interface to receive sentiment predictions. The model will analyze the provided data and return the predicted sentiment along with visualizations of feature importance.

## License

This project is licensed under the MIT License. See the LICENSE file for more details."# multimodal-sentiment-analysis" 
