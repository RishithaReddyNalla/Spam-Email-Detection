import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from flask import Flask, request, render_template

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data.rename(columns={'v1': 'label', 'v2': 'message'})
data = data[['label', 'message']]

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['message'] = data['message'].apply(preprocess)

# Label encoding
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluation
y_pred = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message_processed = preprocess(message)
        message_vectorized = vectorizer.transform([message_processed])
        prediction = model.predict(message_vectorized)[0]

        result = "SPAM" if prediction == 1 else "NOT SPAM"
        return render_template('index.html', prediction=result, original_message=message)

if __name__ == '__main__':
    app.run(debug=True)
