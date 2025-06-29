import pandas as pd
import numpy as np

actions = pd.read_csv('./intermediate/Date_actions.csv')
actions['Date'] = pd.to_datetime(actions['Date'])
actions.set_index('Date', inplace=True)
actions.drop(columns=['Profit','Close'],inplace=True)
print(actions.head())

data = pd.read_csv('./Data/ETHUSDT_1m.csv')
data = data[['Timestamp','Close','High','Low','Open']]
data['Date'] = pd.to_datetime(data['Timestamp'],unit='ms')
data.set_index('Date',inplace= True)
data.drop(columns=['Timestamp'],inplace=True)
data = data.join(actions,how='right')
print(data)

from backtesting import Backtest, Strategy

class ActionStrategy(Strategy):
    def init(self):
        # List of open long entries: (entry_price, entry_bar, order)
        self.open_longs = []
        self.open_shorts = []
        self.bar = 0

    def next(self):
        curr_price = self.data.Close[-1]
        curr_bar = self.bar

        # Check each open long for TP or timeout
        still_open = []
        for entry_price, entry_bar in self.open_longs:
            tp_price = entry_price * 1.02
            held_bars = curr_bar - entry_bar
            if curr_price >= tp_price or held_bars >= 720:
                self.sell(size=1)
            else:
                still_open.append((entry_price, entry_bar))
        self.open_longs = still_open

        still_open = []
        for entry_price, entry_bar in self.open_shorts:
            tp_price = entry_price * 0.98
            held_bars = curr_bar - entry_bar
            if curr_price <= tp_price or held_bars >= 720:
                self.buy(size=1)
            else:
                still_open.append((entry_price, entry_bar))
        self.open_shorts = still_open

        action = self.data.Action[-1]

        if action == 0:
            self.buy(size=1)
            self.open_longs.append((curr_price, curr_bar))

        elif action == 1:
            self.sell(size=1)
            self.open_shorts.append((curr_price, curr_bar))
        
        self.bar+=1
            
# Run the backtest
bt = Backtest(data, ActionStrategy, cash=5000000, commission=0.0002, finalize_trades=True)
stats = bt.run()

print(stats)
bt.plot()
