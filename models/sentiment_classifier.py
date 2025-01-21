from transformers import pipeline
import numpy as np 
import pandas as pd 

from transformers import pipeline
import nltk
import kagglehub
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax



class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze_sentiment(self, text):
        nltk.download('averaged_perceptron_tagger_eng')
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        encoded_text = tokenizer("Truly a sad day", return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2]
        }
        print(scores_dict)

        sent_pipeline = pipeline("sentiment-analysis")
        return sent_pipeline(text)
