# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation

# Load the trained model, vectorizer, and selector
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
selector = pickle.load(open("selector.pkl", "rb"))

# Initialize the Flask app
app = Flask(__name__)

# Download necessary NLTK resources (should be done once)
nltk.download('punkt')
nltk.download('stopwords')

# Define the transform_text function
def transform_text(text):
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum()]
    filtered_words = [word for word in filtered_words if word not in stopwords.words('english') and word not in punctuation]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return " ".join(stemmed_words)

# Define home route to render index.html
@app.route('/')
def home():
    return render_template('index.html')

# Define predict route for POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # Get message data from JSON request
        data = request.get_json()
        msg = data.get("message", "")

        # Preprocess the text
        transformed_msg = transform_text(msg)

        # Vectorize and select features
        vector_input = tfidf.transform([transformed_msg])
        vector_input_selected = selector.transform(vector_input)
        # Predict using the model
        result = model.predict(vector_input_selected)[0]
        prediction = "spam" if result == 1 else "not spam"

        # Return the prediction as a JSON response
        return jsonify({"message": prediction})

if __name__ == "__main__":
    app.run(debug=True)
