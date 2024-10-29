from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load your Keras model
model = tf.keras.models.load_model(r"E:\spam_detectror\textClassifier.keras")

# Load and prepare the dataset (replace with your actual dataset path)
# Ensure you have the actual dataset file path
df = pd.read_csv("E:/spam_detectror/spam-dataset.csv")
X = df['Message']
y = df['Category']

# Train/test split for testing pipeline models (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Create the prediction pipeline using TfidfVectorizer and MultinomialNB
pipeMNB = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', MultinomialNB())])
pipeMNB.fit(X_train, y_train)  # Train pipeline model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = pipeMNB.predict([message])  # Predict with pipeline model
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
