from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define categories
categories = ["Shipping Delay", "Product Quality", "Customer Support", "Payment Issue","Happy Customer", "Constructive Criticism"]

# Classify text
text = "I wish the art had more colour"
result = classifier(text, candidate_labels=categories)

print(result)