## Sentiment analysis from the FinBERT 
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import requests
import pandas as pd 
from bs4 import BeautifulSoup

def get_fed_url(start_date = 2008):
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    headers = {"User-Agent":"Mozilla/5.0"} ## Just letting the website know who is visitng the page just for security purpose

    response = requests.get(url, headers = headers)
    soup = BeautifulSoup(response.text, 'html.parser') ## Extracting out the useful text from the parser

    base_url = "https://www.federalreserve.gov"
    statement_urls = []

    for link in soup.find_all('a', href = True):
        href = link['href']
        if 'monetary' in href and 'pressrelease' in href:
            full_url = base_url + href if href.startswith('/') else href
            statement_urls.append(full_url)
    return statement_urls
    
