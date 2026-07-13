import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.training.train_garch import params_garch
from src.models.garch import Garch
import pandas as pd

def return_value_by_ticker(ticker, datapath = './datas/combined_data.parquet'):
     data = pd.read_par