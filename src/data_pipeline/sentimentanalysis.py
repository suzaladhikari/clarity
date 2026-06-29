## Sentiment analysis from the FinBERT 
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import requests
import pandas as pd 
from bs4 import BeautifulSoup

def get_fed_url(start_date = 2008):
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    headers = {"User-Agent":"Mozilla/5.0"} ## Just letting the website know who is visitng the page just for security purpose

    
