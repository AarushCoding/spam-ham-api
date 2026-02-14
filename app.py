import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)
CORS(app)

# Minimal training data to initialize the model on first run
# You can replace this with a larger .pkl file later
emails = [
    "Get rich quick, click here!", "Free entry to win cash", 
    "Meeting at 5pm today", "Hey, are we still on for lunch?",
    "Congratulations, you won a prize", "Can you send me that report?",
    "URGENT: Your account is locked", "Dinner at my place tonight?"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0] # 1=Spam, 0=Clean

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text_vector = vectorizer.transform([data['text']])
    probs = model.predict_proba(text_vector)[0]
    spam_score = round(probs[1], 3)
    
    verdict = "Spam" if spam_score > 0.5 else "Clean"
    conf = round(max(probs) * 100, 1)
    
    return jsonify({
        'verdict': verdict,
        'score': spam_score,
        'confidence': f"{conf}%"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
