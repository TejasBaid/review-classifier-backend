from flask import Flask, request, jsonify
from models.sentiment_classifier import SentimentAnalyzer
from models.category_classifier import ZeroShotCategoryClassifier

app = Flask(__name__)

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
category_classifier = ZeroShotCategoryClassifier()

@app.route('/')
def home():
    return "Welcome to the Sentiment and Zero-Shot Category Classification API!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Parse input JSON
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "Text field is required"}), 400

        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze_sentiment(text)

        # Category classification
        classification_result = category_classifier.classify(text)

        # Response
        response = {
            "text": text,
            "sentiment": sentiment_result,
            "classification": classification_result["category"],
            "confidence": classification_result["confidence"],
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
