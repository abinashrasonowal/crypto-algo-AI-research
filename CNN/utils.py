import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df, scaler

from sklearn.preprocessing import StandardScaler

def standerd_scale(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
    return df_scaled, scaler


def read_data(tf):
    df = pd.read_csv(f'./Data/BTCUSDT_{tf}.csv')
    df = df[['Timestamp','Open','High','Low','Close','Volume','Taker_buy_base']]
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Date',inplace=True)
    df.drop(columns=['Timestamp'],inplace=True)
    return df


import pandas as pd

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

def add_indicators(df):
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

    return df


def add_target_column(df, trail_mult=1.5, profit_threshold=0.05, lookahead=50):
    target = []

    for i in range(len(df)):
        entry_price = df['Close'].iloc[i]
        sl_long = entry_price - df['atr'].iloc[i] * trail_mult
        sl_short = entry_price + df['atr'].iloc[i] * trail_mult
        profit_long = 0
        profit_short = 0
        valid_bars = df.iloc[i+1:i+1+lookahead]

        # Simulate long trade
        for j in range(len(valid_bars)):
            curr_price = valid_bars['Close'].iloc[j]
            new_sl = curr_price - valid_bars['atr'].iloc[j] * trail_mult
            sl_long = max(sl_long, new_sl)
            profit_long =  (sl_long - entry_price) / entry_price
            if curr_price <= sl_long:
                break
            
        # Simulate short trade
        for j in range(len(valid_bars)):
            curr_price = valid_bars['Close'].iloc[j]
            new_sl = curr_price + valid_bars['atr'].iloc[j] * trail_mult
            sl_short = min(sl_short, new_sl)
            profit_short = (entry_price - sl_short) / entry_price
            if curr_price >= sl_short:
                break
            

        # Assign target
        if profit_long >= profit_threshold:
            target.append(1)
        elif profit_short >= profit_threshold:
            target.append(2)
        else:
            target.append(0)

    df_copy = df.copy()
    df_copy['Target'] = target
    df_target = df_copy[['Target']]
    return df_target
