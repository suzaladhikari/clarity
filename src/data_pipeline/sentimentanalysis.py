## Sentiment analysis from the FinBERT 
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import requests

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert") ## Used to tokenize 
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert") ## For classification of the tokens
sentiment_pipeline = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer) ## Pipeline to do the process

def fetch_news(ticker, start_date, end_date, api_key):
    url = "https://api.tiingo.com/tiingo/news"
    params = {
        "tickers":ticker,
        "startDate": start_date, 
        "endDate":end_date, 
        "token": api_key
    }
    response = requests.get(url, params= params)
    return response.json()

def score_headline(headline, threshold = 0.70):
    result = sentiment_pipeline(headline[:512])[0]
    label = result['label']
    score = result['score']

    if score < threshold: 
        return 0.0 
    
    if label == 'positive':
        return score 
    
    elif label == 'negative':
        return -score
    
    else:
        return 0.0
    


