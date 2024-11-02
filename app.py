# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation
import os
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Define paths to model files
tfidf_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
selector_path = os.path.join(os.path.dirname(__file__), "selector.pkl")

# Load the trained model, vectorizer, and selector with error handling
try:
    tfidf = pickle.load(open(tfidf_path, "rb"))
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    tfidf = None

try:
    model = pickle.load(open(model_path, "rb"))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    selector = pickle.load(open(selector_path, "rb"))
    print("Selector loaded successfully.")
except Exception as e:
    print(f"Error loading selector: {e}")
    selector = None

# Download necessary NLTK resources (should be done once)
nltk.data.path.append('./nltk_data')

# Check if 'punkt' and 'stopwords' resources exist; if not, download them
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='./nltk_data')

# Define the transform_text function
def transform_text(text):
    try:
        text = text.lower()
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.isalnum()]
        filtered_words = [word for word in filtered_words if word not in stopwords.words('english') and word not in punctuation]
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        return " ".join(stemmed_words)
    except Exception as e:
        print(f"Error in transform_text: {e}")
        return ""

# Define home route to render index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define predict route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get message data from JSON request
        data = request.get_json()
        if not data or "message" not in data:
            print("Invalid input: 'message' key is missing.")
            return jsonify({"error": "Invalid input, 'message' key is missing."}), 400

        msg = data.get("message", "")

        # Preprocess the text
        transformed_msg = transform_text(msg)
        if not transformed_msg:
            print("Error during text transformation.")
            return jsonify({"error": "Error during text transformation."}), 500

        # Check if model, tfidf, and selector are loaded
        if tfidf is None or model is None or selector is None:
            print("Model or vectorizer or selector is not loaded.")
            return jsonify({"error": "Model, vectorizer, or selector failed to load."}), 500

        # Vectorize and select features
        try:
            vector_input = tfidf.transform([transformed_msg])
            vector_input_selected = selector.transform(vector_input)
        except Exception as e:
            print(f"Error during vectorization/feature selection: {e}")
            return jsonify({"error": "Error during vectorization/feature selection."}), 500

        # Predict using the model
        try:
            result = model.predict(vector_input_selected)[0]
            prediction = "spam" if result == 1 else "not spam"
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return jsonify({"error": "Error during model prediction."}), 500

        # Return the prediction as a JSON response
        return jsonify({"message": prediction})
    
    except Exception as e:
        # Log any unexpected errors
        print(f"Unexpected error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
