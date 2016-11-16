import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import dateutil
from data_collector import EquityData

class BoardGame(object):
    def __init__(self, symbol="SPY", window=100, start_date=None, end_date=None):
        self.window_size = window
        ed = EquityData(symbol=symbol, start_date=start_date, end_date=end_date)
        hist = ed.collect_history()
        self.board = ed.normalized()
