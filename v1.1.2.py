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

symbols = pd.read_csv('http://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')['Symbol']

class PortfolioGame(object):
    def __init__(self, max_loss=0.75, symbol=None, starting_capital=10000, start_date='2000-01-01', end_date=None,
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
        self.history = np.zeros((self.data.shape[0], 13))
        self.state_shape = self.window * self.history.shape[1]

    def random_symbol(self):
        return choice(symbols)

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
        if self.symbol:
            if self._col:
                self.data = DataReader(self.symbol, 'yahoo', start=start_date, end=end_date)[self._col].to_frame()
            else:
                self.data = DataReader(self.symbol, 'yahoo', start=start_date, end=end_date)
        else:
            flag = True
            while flag:
                try:
                    sym = self.random_symbol()
                    if self._col:
                        self.data = DataReader(sym, 'yahoo', start=start_date, end=end_date)[self._col].to_frame()
                    else:
                        self.data = DataReader(sym, 'yahoo', start=start_date, end=end_date)
                    if self.data.shape[0] > 200:
                        self.symbol = sym
                        #print self.symbol, self.data.shape[0]
                        flag = False
                except:
                    print sym, 'cannot be imported'

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
        self.position = self.current_position_value()

    def portfolio_value(self):
        return self.current_position_value() + self.capital

    def position_pnl(self):
        return self.current_position_value() - self.position

    def portfolio_pnl(self):
        return self.portfolio_value() - self.starting_capital

    def history_df(self):
        return pd.DataFrame(self.history, columns=['current_price', 'portfolio_pnl', 'position', 'postion_pnl', 'capital', 'holding', 'action', 'portfolio_value','sortino'], index=self.data.index).apply(lambda x: x.apply(self.precision))

    def flat_window(self):
        return self.history[self.cursor - self.window:self.cursor].reshape(1, self.history.shape[1] * self.window) if self.cursor >= self.window else None

    def sortino(self):
        port_pct_change = np.diff(self.history[:self.cursor,7]) / self.history[0,7]
        price_pct_change = np.diff(self.history[:self.cursor,0]) / self.history[0,0]
        if port_pct_change.shape[0] == 0 or price_pct_change.shape[0] == 0:
            return 1.0
        sortino = (port_pct_change.sum() - price_pct_change.sum())/(port_pct_change[port_pct_change <= 0.0].std())
        if abs(sortino) == np.inf:
            return 1.0
        return sortino * 1000000

    def game_over(self):
        self.capital -= 1000000
        self.holding = 0
        self.position = 0
        print '-- Game Over --'

    def stop_loss(self,action):
        sl = self.portfolio_value() <= self.starting_capital * (1 - self.max_loss)
        if sl:
            action = 1
            self.game_over()
        return sl, action

    def rolling_mean(self, period):
        return self.data[self._col].rolling(window=period).mean()[self.cursor]

    def step(self, action):
        if self.cursor >= self.data.shape[0]-1:
            return False
        stop_loss, action = self.stop_loss(action)
        perform_action = self.actions[action]
        perform_action()
        self.history[self.cursor] = (self.current_price(), self.rolling_mean(5), self.rolling_mean(22), self.rolling_mean(141), self.rolling_mean(252), self.portfolio_pnl(), self.current_position_value(), self.position_pnl(), self.capital, self.holding, action, self.portfolio_value(), self.sortino())
        self.advance_cursor()
        return not stop_loss


def construct_model(window_size, inputs):
    model = Sequential()
    model.add(Dense(150, init='lecun_uniform', input_shape=(window_size*inputs,)))
    model.add(Activation('relu'))
    model.add(Dense(1000, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # reset weights of neural network
    return model


def train_model(epochs=1000, symbol=None,starting_capital=10000, max_loss=0.5, start_date='2000-01-01', end_date='2017-01-01', col='Adj Close', gamma=0.975, window_size=7, inputs=13, buffer=252):
    batchSize =22
    replay = []
    h=0
    epsilon = 1.0
    status = True
    model = construct_model(window_size=window_size, inputs=inputs)
    dim = window_size * inputs
    start = timer()
    for i in range(epochs):
        game = PortfolioGame(symbol=symbol, max_loss=max_loss, starting_capital=starting_capital, start_date=start_date, end_date=end_date, col=col, window=window_size)
        while status:
            # We are in state S
            # Let's run our Q function on S to get Q values for all possible actions
            if game.cursor <= game.window:
                game.step(2)
            else:
                qval = model.predict(game.flat_window(), batch_size=1)
                # choose best action from Q(s,a) values
                if random() < epsilon:
                    action = np.random.randint(0, 3)
                else:
                    action = (np.argmax(qval))
                # Take action, observe new state S'
                game.step(action)
                # Observe reward
                reward = game.history[game.cursor-1, 8] - game.history[game.cursor-2, 8]
                newQ = model.predict(game.flat_window(), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1, 3))
                y[:] = qval[:]
                if reward <= game.history[game.cursor - 1, 8]:  # non-terminal state
                    update = (reward + (gamma * maxQ)) if maxQ else reward
                else:  # terminal state
                    update = reward
                y[0][action] = update
                model.fit(game.flat_window(), y, batch_size=1, nb_epoch=1, verbose=1)

                # Experience replay storage
                if len(replay) < buffer:  # if buffer not filled, add to it
                    replay.append((game.history[((game.cursor - 2) - game.window):game.cursor - 2].reshape(1, dim), action,
                                   reward, game.history[((game.cursor) - game.window):game.cursor].reshape(1, dim)))
                else:  # if buffer full, overwrite old values
                    if h < buffer - 1:
                        h += 1
                    else:
                        h = 0
                    replay[h] = (
                    game.history[((game.cursor - 2) - game.window):game.cursor - 2].reshape(1, dim), action, reward,
                    game.history[((game.cursor) - game.window):game.cursor].reshape(1, dim))
                    # randomly sample our experience replay memory
                    minibatch = sample(replay, batchSize)
                    X_train = []
                    y_train = []
                    for memory in minibatch:
                        # Get max_Q(S',a)
                        old_state, action, reward, new_state = memory
                        old_qval = model.predict(old_state, batch_size=1)
                        newQ = model.predict(new_state, batch_size=1)
                        maxQ = np.max(newQ)
                        y = np.zeros((1, 3))
                        y[:] = old_qval[:]
                        if reward > -game.starting_capital:  # non-terminal state
                            update = (reward + (gamma * maxQ))
                        else:  # terminal state
                            update = reward
                        y[0][action] = update
                        model.fit(old_state, y.reshape(1, 3), batch_size=1, nb_epoch=1, verbose=1)
                        X_train.append(old_state)
                        y_train.append(y.reshape(1, 3))
        if epsilon > 0.05:  # decrement epsilon over time
            epsilon -= (1 / epochs)
        print i, game.symbol, game.cursor, round((game.history[game.cursor - 1, 7] / game.starting_capital) * 100,
                                              2), '%', round(
            (game.history[game.cursor - 1, 0] / game.history[0, 0]) * 100, 2), '%'
        #print timer()-start
        #game.history_df()['portfolio_value'].plot()
        game.history_df().to_csv('dqn_train.csv')
        #print game.history[-1,8]
    return model


def test_model(model, symbol='SPY', starting_capital=10000, max_memory=7, max_loss=0.5, start_date='2000-01-01', end_date=None, col='Adj Close', gamma=0.975, buffer=252):
    batchSize = buffer /4
    replay = []
    h=0
    status = True
    print '===== TEST ====='
    game = PortfolioGame(symbol=symbol, max_loss=max_loss, start_date=start_date,end_date=end_date, col=col, starting_capital=starting_capital, window=max_memory)
    dim = game.history.shape[1] * game.window
    while status:
        # Let's run our Q function on S to get Q values for all possible actions
        if game.cursor <= game.window:
            game.step(2)
        else:
            qval = model.predict(game.flat_window(), batch_size=1)
            action = (np.argmax(qval))  # choose best action from Q(s,a) values
            # Take action, observe new state S'
            game.step(action)
            # Observe reward
            reward = game.history[game.cursor-1, 8] - game.history[game.cursor-2, 8]
            # Get max_Q(S',a)
            newQ = model.predict(game.flat_window(), batch_size=1)
            maxQ = np.max(newQ)
            y = np.zeros((1, 3))
            y[:] = qval[:]
            if reward <= game.history[game.cursor-1, 8]:  # non-terminal state
                update = (reward + (gamma * maxQ))
            else:  # terminal state
                update = reward
            y[0][action] = update
            model.fit(game.flat_window(), y, batch_size=1, nb_epoch=1, verbose=1)
            """
            # Experience replay storage
            if len(replay) < buffer:  # if buffer not filled, add to it
                replay.append((game.history[((game.cursor - 2) - game.window):game.cursor - 2].reshape(1, dim), action,
                               reward, game.history[((game.cursor) - game.window):game.cursor].reshape(1, dim)))
            elif random() < 0.5:  # if buffer full, overwrite old values
                if h < buffer - 1:
                    h += 1
                else:
                    h = 0
                replay[h] = (game.history[((game.cursor - 2) - game.window):game.cursor - 2].reshape(1, dim), action, reward,
                             game.history[((game.cursor) - game.window):game.cursor].reshape(1, dim))
                # randomly sample our experience replay memory
                minibatch = sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    # Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    old_qval = model.predict(old_state, batch_size=1)
                    newQ = model.predict(new_state, batch_size=1)
                    maxQ = np.max(newQ)
                    y = np.zeros((1, 3))
                    y[:] = old_qval[:]
                    if reward > -game.starting_capital:  # non-terminal state
                        update = (reward + (gamma * maxQ))
                    else:  # terminal state
                        update = reward
                    y[0][action] = update
                    model.fit(old_state, y.reshape(1, 3), batch_size=1, nb_epoch=1, verbose=1)
                    X_train.append(old_state)
                    y_train.append(y.reshape(1, 3))
                replay = []
            else:
                replay = []
        """
    game.history_df().to_csv('dqn_test.csv')
    print game.history_df()

if __name__ == '__main__':
    starting_capital = 25000
    symbol = 'SPY'
    train_start_date = '2010-01-01'
    train_end_date = '2018-07-01'
    epochs = 25
    max_memory = 7
    max_loss = 0.75
    model = train_model(epochs=epochs,symbol=symbol, starting_capital=starting_capital, window_size=max_memory,
                        max_loss=max_loss, start_date=train_start_date, end_date=train_end_date, buffer=252)

    test_model(model, symbol=symbol, starting_capital=starting_capital, start_date=train_end_date, max_loss=max_loss, max_memory=max_memory)