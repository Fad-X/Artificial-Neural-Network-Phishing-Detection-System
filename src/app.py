from flask import Flask, request, jsonify, render_template
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from utils import preprocess_data
import numpy as np
import pandas as pd
import pickle
import os
import io
from contextlib import redirect_stdout

app = Flask(__name__, template_folder='../templates')

MODEL_PATH = './model/phishing_ann_model.h5'
VECTORIZER_PATH = './model/vectorizer.pkl'
DATA_PATH = './data/combined_url_data.csv'

model = None
vectorizer = None

def load_model_and_vectorizer():
    global model, vectorizer
    model = load_model(MODEL_PATH)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        input_data = preprocess_data([url], vectorizer)
        input_data = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data)
        result = 'Legitimate' if prediction[0][0] <= 0.4 else 'Phishing'

        return jsonify({'url': url, 'result': result})

    except Exception as e:
        print(f'Prediction Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    logs = io.StringIO()
    try:
        data = pd.read_csv(DATA_PATH)
        X_urls = data['url'].values
        y = data['label'].values

        vectorizer_local = CountVectorizer()
        X_processed = vectorizer_local.fit_transform(X_urls).toarray()

        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, stratify=y, random_state=42
        )

        model_local = Sequential()
        model_local.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
        model_local.add(Dropout(0.5))
        model_local.add(Dense(units=32, activation='relu'))
        model_local.add(Dropout(0.5))
        model_local.add(Dense(units=1, activation='sigmoid'))

        model_local.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        with redirect_stdout(logs):
            model_local.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=3)
            model_local.fit(
                X_train, y_train, batch_size=32, epochs=50,
                validation_data=(X_val, y_val), callbacks=[early_stopping]
            )

        model_local.save(MODEL_PATH)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer_local, f)

        load_model_and_vectorizer()

        training_output = logs.getvalue()
        return jsonify({'message': 'Training completed.', 'details': training_output})

    except Exception as e:
        error_message = str(e)
        print(f'Training Error: {error_message}')
        return jsonify({'error': error_message}), 500

    finally:
        logs.close()

if __name__ == '__main__':
    try:
        load_model_and_vectorizer()
    except Exception as e:
        print(f"Failed to load model/vectorizer initially: {e}")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
