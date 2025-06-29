import pandas as pd

df = pd.read_csv('./Data/ETHUSDT_1h.csv')
df= df[['Timestamp','Close','High','Low','Open','Volume', 'Taker_buy_base']]
df['Date'] = pd.to_datetime(df['Timestamp'],unit='ms')
df.set_index(df['Date'],inplace= True)
df.drop(columns=['Timestamp'],inplace=True)
df.drop(columns=['Date'],inplace=True)
print(df.head())

from ta.trend import (
    SMAIndicator, MACD, CCIIndicator, ADXIndicator, TRIXIndicator,
    WMAIndicator,PSARIndicator
)
from ta.momentum import (
    ROCIndicator, StochasticOscillator, StochRSIIndicator,
    UltimateOscillator, AwesomeOscillatorIndicator
)
from ta.volume import (
    MFIIndicator,OnBalanceVolumeIndicator
)
from ta.volatility import (
    AverageTrueRange,BollingerBands
)
from ta.others import (
    DailyReturnIndicator
)


# Ensure proper capitalization
df.columns = df.columns.str.capitalize()

# Add Indicators

df['Delta_volume'] = df['Taker_buy_base'] - (df['Volume'] - df['Taker_buy_base'])

# Trend Indicators
df['SMA_26'] = SMAIndicator(close=df['Close'], window=26).sma_indicator()
df['SMA_55'] = SMAIndicator(close=df['Close'], window=55).sma_indicator()
# df['macd'] = MACD(close=df['Close']).macd()
# df['cci'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
df['adx'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
# df['trix'] = TRIXIndicator(close=df['Close']).trix()
# df['tema'] = TEMAIndicator(close=df['Close']).tema_indicator()
# df['trima'] = ta.trend.TRIMAIndicator(close=df['Close']).trima_indicator()
# df['wma'] = WMAIndicator(close=df['Close']).wma()
# df['dema'] = DEMAIndicator(close=df['Close']).dema_indicator()
# df['sar'] = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close']).psar()

# Momentum Indicators
# df['roc'] = ROCIndicator(close=df['Close']).roc()
# df['stoch_k'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
# df['stoch_d'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch_signal()
# df['cmo'] = CMOIndicator(close=df['Close']).cmo()
# df['stochrsi'] = StochRSIIndicator(close=df['Close']).stochrsi()
# df['uo'] = UltimateOscillator(high=df['High'], low=df['Low'], close=df['Close']).ultimate_oscillator()
# df['ao'] = AwesomeOscillatorIndicator(high=df['High'], low=df['Low']).awesome_oscillator()

# Volume Indicators
df['obv'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
# df['mfi'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).money_flow_index()
# df['bop'] = BalanceOfPowerIndicator(open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']).balance_of_power()

# Volatility
df['atr'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
indicator_bb = BollingerBands(close=df["Close"])
# df['bb_bbm'] = indicator_bb.bollinger_mavg()
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()


# Other Indicators
# df['daily_return'] = DailyReturnIndicator(close=df['Close']).daily_return()

# Final cleaning
df.dropna(inplace=True)  # Remove NaNs from early periods of indicators

# df['lower_wick'] = df[['Close', 'Open']].min(axis=1) -  df['Low']
# df['upper_wick'] = df['High'] - df[['Close', 'Open']].max(axis=1)
df['body_size'] = df['Close'] - df['Open']
df.drop(columns=['High','Open','Low','Volume','Taker_buy_base'],inplace=True)

print(df.shape)

print(df.info())

df.to_csv('./intermediate/ETHUSDT_1h_indicators.csv',index=True)
