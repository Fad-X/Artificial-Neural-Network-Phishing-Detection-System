# ANN-Based Phishing Detection System

## Overview

This project aims to develop a phishing detection system based on an Artificial Neural Network (ANN). It classifies URLs as either legitimate or phishing based on patterns in the URLs themselves. The system is trained using a dataset of legitimate and phishing URLs, and the trained model is used to make predictions on new URLs.

## Features

- **Artificial Neural Network (ANN):** The core of the system uses an ANN model to detect phishing URLs.
- **Preprocessing:** The system preprocesses URLs using `CountVectorizer` to convert them into a format suitable for training the ANN model.
- **Binary Classification:** Predicts whether a given URL is `Legitimate` or `Phishing`.
- **Custom Dataset:** Uses both phishing data from [OpenPhish](https://openphish.com/) and legitimate data scraped from trusted websites.
- **Interactive Prediction:** Allows users to input URLs and get immediate predictions on whether the URL is safe or phishing.

## Directory Structure

ANN-Phishing-Detection-System/ │ ├── model/ # Directory containing the trained model and vectorizer │ ├── phishing_ann_model.h5 │ └── vectorizer.pkl ├── src/ # Contains the source code for training and prediction │ ├── train_model.py # Script to train the ANN model │ ├── predict.py # Script to predict if a URL is phishing or legitimate │ ├── utils.py # Helper functions for preprocessing │ └── requirements.txt # Python package dependencies ├── datasets/ # Contains training data │ ├── phishing_data.csv │ ├── legitimate_data.csv │ └── combined_data.csv # Combined dataset of both phishing and legitimate URLs ├── README.md # This readme file └── .gitignore # Files to be ignored by git

perl
Copy code

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
Main dependencies include:

TensorFlow: Used for building and training the ANN model.
Scikit-learn: For data preprocessing and vectorization.
Numpy: For numerical operations.
Pandas: For data manipulation.
Getting Started
1. Clone the repository
bash
Copy code
git clone https://github.com/Fad-X/Artificial-Neural-Network-Phishing-Detection-System.git
cd ANN-Phishing-Detection-System
2. Create and activate a virtual environment (Optional but recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Train the model
If you want to train the ANN model from scratch using the provided dataset, run:

bash
Copy code
python src/train_model.py
This script will preprocess the URLs, vectorize them, and train the ANN model. After training, it will save the model (phishing_ann_model.h5) and the vectorizer (vectorizer.pkl) in the model/ directory.

5. Make predictions
To predict if a URL is phishing or legitimate, run:

bash
Copy code
python src/predict.py
The script will ask for a URL, make predictions, and output whether the URL is classified as Phishing or Legitimate.

Looping URL predictions
The predict.py script is designed to loop and continuously ask for URLs to predict until you manually stop it (by pressing Ctrl + C).

6. Dataset
Phishing URLs: Data is sourced from OpenPhish.
Legitimate URLs: Legitimate website data is scraped from trusted sites such as Google, Wikipedia, and others.
The datasets used in this project are found in the datasets/ directory:

phishing_data.csv: Contains phishing URLs.
legitimate_data.csv: Contains legitimate URLs.
combined_data.csv: Merged dataset of phishing and legitimate URLs used for training.
Model Architecture
The ANN model consists of three fully connected layers with dropout regularization to avoid overfitting:

Input Layer: Vectorized URL data.
Hidden Layer 1: 64 units with ReLU activation and dropout.
Hidden Layer 2: 32 units with ReLU activation and dropout.
Output Layer: Single unit with sigmoid activation for binary classification (Phishing or Legitimate).
Results
After training the model on the combined dataset, it achieved the following performance metrics:

Training Accuracy: 99.8%
Validation Accuracy: 98%
Validation Loss: 0.045
These results indicate that the model performs well in distinguishing phishing URLs from legitimate ones.

Future Improvements
Integrate real-time URL checking.
Incorporate more features for improved accuracy.
Expand dataset with more diverse sources.
Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to submit a pull request or open an issue.
