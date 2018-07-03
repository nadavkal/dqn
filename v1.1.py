from random import choice,random,sample
from IPython.display import clear_output
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from pandas_datareader.data import DataReader
import datetime
import numpy as np
import pandas as pd
from timeit import default_timer as timer


class PortfolioGame(object):
    def __init__(self, max_loss=0.75, symbol='AAPL', starting_capital=10000, start_date='2000-01-01', end_date=None,
                 col='Adj Close', window=7):
        self.window = window
        self.max_loss = max_loss
        self._col = col
        self.starting_capital = starting_capital
        self.capital = self.starting_capital
        self.symbol = symbol
        self.data = None
        self.read_history(start_date, end_date)
        self.steps = self.data.shape[0]
        self.cursor = 0
        self.holding = 0
        self.position = 0
        self.actions = {0: self.buy, 1: self.sell, 2: self.hold}
        self.history = np.zeros((self.data.shape[0], 8))
        self.state_shape = self.window * self.history.shape[1]

    def format_date(self, date):
        if isinstance(date, str):
            return date
        elif isinstance(date, datetime.date):
            return date.isoformat()
        elif date is None:
            return datetime.date.today().isoformat()
        else:
            raise ('Date Error in Portfolio Game')

    def read_history(self, start_date, end_date):
        start_date, end_date = self.format_date(start_date), self.format_date(end_date)
        if self._col:
            self.data = DataReader(self.symbol, 'yahoo', start=start_date, end=end_date)[self._col].to_frame()
        else:
            self.data = DataReader(self.symbol, 'yahoo', start=start_date, end=end_date)

    def advance_cursor(self):
        self.cursor += 1

    def precision(self, x):
        return round(x, 4)

    def current_price(self):
        # print self.cursor, self._col, self.data.shape
        return self.precision(self.data.iloc[self.cursor][self._col])

    def calc_max_buy(self):
        return int(self.capital / self.current_price())

    def position_buy_value(self):
        return self.precision(self.calc_max_buy() * self.current_price())

    def current_position_value(self):
        return self.holding * self.current_price()

    def buy(self):
        self.holding += int(self.calc_max_buy())
        pos_value = self.position_buy_value()
        self.capital -= pos_value
        self.position += pos_value

    def sell(self):
        value = self.current_position_value()
        self.capital += value
        self.position = 0
        self.holding -= value / self.current_price()

    def hold(self):
        pass

    def portfolio_value(self):
        return self.position + self.capital

    def position_pnl(self):
        return self.current_position_value() - self.position

    def portfolio_pnl(self):
        return (self.capital + self.current_position_value()) - self.starting_capital

    def history_df(self):
        df = pd.DataFrame(self.history,
                          columns=['current_price', 'portfolio_pnl', 'portfolio_value', 'position', 'postion_pnl',
                                   'capital', 'holding', 'action'], index=self.data.index)
        # df['action_type'] = df.action.apply(lambda x: self.actions[x])
        return df.apply(lambda x: x.apply(self.precision))

    def flat_window(self):
        if self.cursor >= self.window:
            return self.history[self.cursor - self.window:self.cursor].reshape(1, self.history.shape[1] * self.window)
        else:
            return None

    def game_over(self):
        self.capital -= 1000000
        self.holding = 0
        self.position = 0
        print '-- Game Over --'

    def step(self, action):
        if self.portfolio_value() <= self.starting_capital * (1 - self.max_loss):
            action = 1
            if self.portfolio_value() < -10000:
                self.game_over()
                action = 2
        perform_action = self.actions[action]
        perform_action()
        self.history[self.cursor] = (
        self.current_price(), self.portfolio_pnl(), self.portfolio_value(), self.position, self.position_pnl(),
        self.capital, self.holding, action)
        self.advance_cursor()


epochs = 1000
gamma = 0.975
epsilon = 1
batchSize = 25
buffer = 50
replay = []

model = Sequential()
model.add(Dense(150, init='lecun_uniform', input_shape=(56,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(1000, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(3, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.compile(loss='mse', optimizer=rms)  # reset weights of neural network
# stores tuples of (S, A, R, S')
h = 0
start = timer()
for i in range(epochs):
    game = PortfolioGame(max_loss=0.5, start_date='2000-01-01', end_date='2017-01-01', col='Adj Close')
    while game.cursor < game.steps-1:
        # We are in state S
        # Let's run our Q function on S to get Q values for all possible actions
        if game.cursor <= game.window:
            game.step(2)
        else:
            qval = model.predict(game.flat_window(), batch_size=1)
            if random() < epsilon:
                action = np.random.randint(0, 3)
            else:
                action = (np.argmax(qval))  # choose best action from Q(s,a) values
            # Take action, observe new state S'
            game.step(action)
            # Observe reward
            reward = game.history[game.cursor, 1] - game.history[game.cursor-1, 1]
            # Get max_Q(S',a)
            newQ = model.predict(game.flat_window(), batch_size=1)
            maxQ = np.max(newQ)
            y = np.zeros((1, 3))
            y[:] = qval[:]
            if reward > -game.starting_capital:  # non-terminal state
                update = (reward + (gamma * maxQ))
            else:  # terminal state
                update = reward
            y[0][action] = update
            model.fit(game.flat_window(), y, batch_size=batchSize, nb_epoch=1, verbose=1)
        #print 'TTG:%.2f minutes' % ((timer()-start)/(i+1))*((epochs-(i+1))/60)
        #clear_output(wait=True)
    if epsilon > 0.05:  # decrement epsilon over time
        epsilon -= (1 / epochs)
    game.history_df().to_csv('dqn.csv')
print timer()-start
#game.history_df()['portfolio_value'].plot()


print '===== TEST ====='
game = PortfolioGame(symbol='VIX', max_loss=0.5, start_date='2016-12-01', col='Adj Close')
while game.cursor < game.steps-1:
    # We are in state S
    # Let's run our Q function on S to get Q values for all possible actions
    if game.cursor <= game.window:
        game.step(2)
    else:
        qval = model.predict(game.flat_window(), batch_size=1)
        action = (np.argmax(qval))  # choose best action from Q(s,a) values
        # Take action, observe new state S'
        game.step(action)
        # Observe reward
        reward = game.history[game.cursor, 1] - game.history[game.cursor-1, 1]
        # Get max_Q(S',a)
        newQ = model.predict(game.flat_window(), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1, 3))
        y[:] = qval[:]
        if reward > -game.starting_capital:  # non-terminal state
            update = (reward + (gamma * maxQ))
        else:  # terminal state
            update = reward
        y[0][action] = update
        model.fit(game.flat_window(), y, batch_size=batchSize, nb_epoch=1, verbose=1)
    #print 'TTG:%.2f minutes' % ((timer()-start)/(i+1))*((epochs-(i+1))/60)
print game.history_df()