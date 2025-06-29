import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def min_max_scal(df):
    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    return scaled_df, scaler
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def visualize_actions(Date_actions):
    prices = Date_actions['Close'].values
    actions = Date_actions['Action'].values
    cum_profit = Date_actions['Profit'].values

    buy_steps = [i for i, a in enumerate(actions) if a == 0]
    sell_steps = [i for i, a in enumerate(actions) if a == 1]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Top Plot: Price with Actions ---
    axs[0].plot(prices, label="Price", color="blue", alpha=0.7)
    axs[0].scatter(buy_steps, [prices[i] for i in buy_steps], color="green", label="Buy", marker="^", alpha=0.8)
    axs[0].scatter(sell_steps, [prices[i] for i in sell_steps], color="red", label="Sell", marker="v", alpha=0.8)
    axs[0].set_title("Actions Taken by the Agent")
    axs[0].set_ylabel("Price")
    axs[0].legend()
    axs[0].grid(True)

    # --- Bottom Plot: Cumulative Profit ---
    axs[1].plot(cum_profit, label="Cumulative PnL", color="green", alpha=0.8)
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Cumulative Profit")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('output.jpg')
    plt.show()


    
