# Code Written by Fad-X
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def load_data(data_path):
    return pd.read_csv(data_path)

def preprocess_data(X_urls, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()  # Create a new vectorizer if none is provided
        X_processed = vectorizer.fit_transform(X_urls).toarray()
    else:
        X_processed = vectorizer.transform(X_urls).toarray()  # Use the provided vectorizer
    return X_processed, vectorizer

# Step 1: Load and preprocess the data
data_path = 'data_path = '../data/combined_url_data.csv'
X = load_data(data_path)
X_urls = X['url'].values
y = X['label'].values

# Preprocess the URLs and get the vectorizer
X_processed, vectorizer = preprocess_data(X_urls)

# Step 2: Split the data into training and validation sets (stratified)
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

# Step 3: Build the ANN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Step 5: Save the model and vectorizer
model.save(os.path.join('..', 'model', 'phishing_ann_model.h5'))
with open(os.path.join('..', 'model', 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")
