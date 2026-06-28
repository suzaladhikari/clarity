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
