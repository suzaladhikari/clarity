## Sentiment analysis from the FinBERT 
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import requests
import pandas as pd 
from ingest import watchlist

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert") ## Used to tokenize 
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert") ## For classification of the tokens
sentiment_pipeline = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer) ## Pipeline to do the process

def fetch_news(ticker, start_date, api_key):
    url = "https://api.tiingo.com/tiingo/news"
    params = {
        "tickers":ticker,
        "startDate": start_date, 
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
    
def build_daily_sentiment(ticker, start_date, api_key):
    articles = fetch_news(ticker, start_date, api_key)
    if not articles:
        return pd.DataFrame(columns = ['date', 'sentiment'])

    records = []
    for article in articles:
        ## Storing the date of the article published
        date = pd.to_datetime(article['publishedDate']).date() ## date of published
        headline = article.get('title', '') ## Storing the title only, thats all we need
        score = score_headline(headline) ## Calculating the score
        records.append({'date':date, 'sentiment': score})

    data = pd.DataFrame(records)

    ## Averaging each day's headlines scores and making it one score per day 
    daily = data.groupby('date')['sentiment'].mean().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])

    return daily


all_sentiments = []

for stock in watchlist:
    stockData = build_daily_sentiment(stock, '2008-01-01','8e14d5babfe29c8815f268eb1afa1727ce18f16e' )
    all_sentiments.append(stockData)

sentiment_data = pd.concat(all_sentiments, ignore_index=True)
print(sentiment_data.shape[0])
