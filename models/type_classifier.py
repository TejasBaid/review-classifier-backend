from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

class EnhancedKeywordClassifier:
    def __init__(self):
        self.categories = {
            "Shipping Delay": ["late delivery", "delayed shipment", "shipping issue"],
            "Product Quality": ["damaged product", "defective item", "poor quality"],
            "Customer Support": ["bad service", "unhelpful support", "no response"],
            "Payment Issue": ["payment failed", "refund issue", "billing problem"]
        }
        self.fuzzy_threshold = 80
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.category_embeddings = self._compute_category_embeddings()

    def _compute_category_embeddings(self):
        # Compute embeddings for category phrases
        embeddings = {}
        for category, phrases in self.categories.items():
            embeddings[category] = self.semantic_model.encode(phrases, convert_to_tensor=True)
        return embeddings

    def classify(self, text):
        # Semantic similarity
        text_embedding = self.semantic_model.encode(text, convert_to_tensor=True)
        best_category = None
        best_similarity = 0.0

        for category, embeddings in self.category_embeddings.items():
            similarity_scores = util.cos_sim(text_embedding, embeddings).max().item()
            if similarity_scores > best_similarity:
                best_similarity = similarity_scores
                best_category = category

        # Fuzzy matching
        fuzzy_results = {}
        text_lower = text.lower()
        for category, phrases in self.categories.items():
            fuzzy_scores = [fuzz.partial_ratio(text_lower, phrase) for phrase in phrases]
            fuzzy_results[category] = max(fuzzy_scores)

        best_fuzzy_match = max(fuzzy_results, key=fuzzy_results.get)
        if fuzzy_results[best_fuzzy_match] >= self.fuzzy_threshold:
            return best_fuzzy_match if best_similarity < 0.7 else best_category  # Prioritize semantic

        # Final decision: prioritize semantic if similarity is high
        return best_category if best_similarity > 0.5 else "Unclassified"
