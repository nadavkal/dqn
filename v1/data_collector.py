import pandas as pd
import numpy as np
from datetime import date
from dateutil.parser import parse as dateparser
import ystockquote


def date_to_str(d):
    if type(d) == 'NoneType':
        raise Exception('could not convert None date to string')
    if type(d) == 'str':
        return d
    if type(d) == 'unicode':
        return str(d)
    if type(d) == 'datetime.date':
        return d.strftime('%Y-%m-%d')


class EquityData(object):
    def __init__(self, symbol="SPY", start_date=None, end_date=None):
        self.symbol = symbol
        self.start_date = start_date if start_date else "1980-01-01"
        self.start_date = date_to_str(self.start_date)
        self.end_date = end_date if end_date else date.today().strftime('%Y-%m-%d')
        self.history = None
        self.normalized = None

    def collect_history(self):
        if self.end_date:
            df = pd.DataFrame(ystockquote.get_historical_prices(
                symbol=self.symbol, start_date=self.start_date, end_date=self.end_date)).T
        else:
            df = pd.DataFrame(ystockquote.get_historical_prices(
                symbol=self.symbol, start_date=self.start_date)).T
        df = df.apply(lambda x: pd.to_numeric(x), axis=1)
        self.history = df
        return df

    def normalize_price(self, df=None, column="Adj Close"):
        if not (self.history or df):
            raise Exception("Could not normalize price,no history for equity %s" %self.symbol)
        self.normalized = df[column] if df else self.history[column]
        self.normalized = self.normalized.pct_change()[1:]
        return self.normalized

    def collect_firmo(self):
        return ystockquote.get_all(self.symbol)
