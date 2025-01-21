from transformers import pipeline

class ZeroShotCategoryClassifier:
    def __init__(self):
        # Initialize the zero-shot classification pipeline
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Define comprehensive categories
        self.categories = [
            "Shipping Delay",
            "Product Quality",
            "Customer Support",
            "Payment Issue",
            "Happy Customer",
            "Constructive Criticism",
            "Feature Request",
            "Technical Problem",
            "Pricing Issue",
            "Usability Concern",
            "Return/Exchange Issue",
            "Delivery Experience",
        ]

    def classify(self, text):
        # Perform zero-shot classification
        result = self.classifier(text, candidate_labels=self.categories)

        # Extract the best category and its confidence score
        best_category = result["labels"][0]
        confidence = result["scores"][0]

        return {"category": best_category, "confidence": confidence}
