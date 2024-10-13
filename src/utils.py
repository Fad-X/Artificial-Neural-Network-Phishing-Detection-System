# Code Written by Fad-X
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def load_data(data_path):
    # Load data from CSV
    return pd.read_csv(data_path)

# Load and preprocess the data
data_path = os.path.join('..', 'data', 'phishing_data.csv')
X = load_data(data_path)

# Debugging: Check the structure of the loaded data
print("Data type of X:", type(X))  # Check the type of X
print("Data shape:", X.shape)  # Print shape of the DataFrame
#print("Data preview:\n", X.head())  # Preview the first few rows

# Check if 'url' column exists
if isinstance(X, pd.DataFrame) and 'url' in X.columns and 'label' in X.columns:
    # Extract URLs and labels
    X_urls = X['url'].values  # Get URLs as a numpy array
    y = X['label'].values  # Get labels as a numpy array
else:
    raise ValueError("The loaded data is not a DataFrame or does not contain the expected columns 'url' and 'label'")


from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(X_urls, vectorizer=None):
    """
    Preprocess the URLs into numerical format using the CountVectorizer.
    If vectorizer is provided, it will be used. Otherwise, a new vectorizer is created and fitted.
    """
    if vectorizer is None:
        # Create a new vectorizer (for training)
        vectorizer = CountVectorizer()
        X_processed = vectorizer.fit_transform(X_urls).toarray()
    else:
        # Use the provided vectorizer (for prediction)
        X_processed = vectorizer.transform(X_urls).toarray()
    
    return X_processed

def split_data(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
