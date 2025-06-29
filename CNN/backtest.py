import pandas as pd

actions = pd.read_csv('./intermediate/test_results.csv')
actions['Date'] = pd.to_datetime(actions['Date'])
actions.set_index('Date', inplace=True)
actions.drop(columns=['Actual'],inplace=True)
print(actions['Predicted'].value_counts())
# print(actions.head())

data = pd.read_csv('./Data/BTCUSDT_1d.csv')
data = data[['Timestamp','Close','High','Low','Open']]
data['Date'] = pd.to_datetime(data['Timestamp'],unit='ms')
data.set_index('Date',inplace= True)
data.drop(columns=['Timestamp'],inplace=True)
data = data.join(actions,how='right')
# print(data.head())
# exit()

from backtesting import Backtest, Strategy
class ActionStrategy(Strategy):
    def init(self):
        self.bar = 0
        self.entry_bar = []

    def next(self):        
        prediction = self.data.Predicted[self.bar]

        if prediction == 1:
            self.buy(size=1)
            self.entry_bar.append(self.bar)

        i = 0
        while i < len(self.entry_bar):
            entry = self.entry_bar[i]
            if self.bar - entry >= 24 or self.data.Close[self.bar] >= self.data.Close[entry] * 1.05:
                if self.position:
                    self.position.close()
                self.entry_bar[i], self.entry_bar[-1] = self.entry_bar[-1], self.entry_bar[i]
                self.entry_bar.pop()
            else:
                i += 1

        self.bar += 1
        
# Run the backtest
bt = Backtest(data, ActionStrategy, cash=1800000, commission=0.0002, finalize_trades=True)
stats = bt.run()

print(stats)
bt.plot()
