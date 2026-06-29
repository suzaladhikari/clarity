## Sentiment analysis from the FinBERT 
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import requests
import pandas as pd 
from bs4 import BeautifulSoup