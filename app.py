from flask import Flask, request, jsonify
from models.sentiment_classifier import SentimentAnalyzer
from models.type_classifier import EnhancedKeywordClassifier

app = Flask(__name__)

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
keyword_classifier = EnhancedKeywordClassifier()

@app.route('/')
def home():
    return "Welcome to the Sentiment and Enhanced Keyword Classification API!"

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

        # Keyword classification
        keyword_result = keyword_classifier.classify(text)
        print(keyword_result)
        # Response
        response = {
            "text": text,
            "sentiment": sentiment_result,
            "classification": keyword_result
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
