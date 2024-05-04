"""
By : Sebastian Mora (@Bastian110)
Project : To-Kill-A-Mocking-Bird
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from AbstractSentry import AbstractSentry

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_and_stopwords(doc: str) -> str:
    words = nltk.word_tokenize(doc.lower())
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)

app = Flask("TKAMT")
CORS(app)

sentry = AbstractSentry(stem_and_stopwords, "mongodb://localhost:27017/", "TKAMT")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message" : "This is to Kill a Mocking Text 🩸"})

@app.route("/compare-with-database", methods=["POST"])
def compare_with_database():
    try:
        document = request.get_json()["text"]
        result = sentry.calculate_similarity(document)
        plagiarized = 1 if result[0]["similarity"] > 0.25 else 0
        return jsonify({"message" : "everything okay!", "results" : result, "plagiarized" : plagiarized}), 200
    except Exception as error:
        return jsonify({"message" : "something went wrong"}), 401
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)