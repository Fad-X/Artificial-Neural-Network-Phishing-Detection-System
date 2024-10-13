# Code Written by Fad-X

from keras.models import load_model
from utils import preprocess_data
import numpy as np
import pickle
import os

# Load the trained model and vectorizer
model = load_model('../model/phishing_ann_model.h5')

# Load the vectorizer used during training
with open('../model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def predict_url(url):
    """
    Predict if the input URL is phishing or legitimate.
    """
    # Preprocess the input URL using the loaded vectorizer
    input_data = preprocess_data([url], vectorizer)

    # Ensure input_data has the correct shape (1, -1)
    input_data = np.array(input_data).reshape(1, -1)

    # Predict and return the result
    prediction = model.predict(input_data)
    return 'Phishing' if prediction[0][0] >= 0.5 else 'Legitimate'

if __name__ == "__main__":
    try:
        while True:
            user_url = input("\nEnter a URL to check (Ctrl+C to exit): ")
            result = predict_url(user_url)
            print(f'The URL "{user_url}" is: {result}')
    except KeyboardInterrupt:
        print("\nExiting...")

