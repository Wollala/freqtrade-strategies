"""
Supertrend strategy:
* Description: Generate a 3 supertrend indicators for 'buy' strategies & 3 supertrend indicators for 'sell' strategies
               Buys if the 3 'buy' indicators are 'up'
               Sells if the 3 'sell' indicators are 'down'
* Author: @juankysoriano (Juan Carlos Soriano)
* github: https://github.com/juankysoriano/

* NOTE: This Supertrend strategy is just one of many possible strategies using `Supertrend` as indicator. It should on any case used at your own risk.
      It comes with at least a couple of caveats:
        1. The implementation for the `supertrend` indicator is based on the following discussion: https://github.com/freqtrade/freqtrade-strategies/issues/30.
           Concretelly https://github.com/freqtrade/freqtrade-strategies/issues/30#issuecomment-853042401
        2. The implementation for `supertrend` on this strategy is not validated;
           Meaning this that is not proven to match the results by the paper where it was originally introduced or any other trusted academic resources
"""

import time

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
'''
============================================================ BACKTESTING REPORT ===========================================================
|     Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
|----------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
| ETH/USDT |       201 |           3.06 |         614.61 |        259469.020 |       25946.90 | 3 days, 2:57:00 |   186     0    15  92.5 |
|    TOTAL |       201 |           3.06 |         614.61 |        259469.020 |       25946.90 | 3 days, 2:57:00 |   186     0    15  92.5 |
=========================================================== ENTER TAG STATS ============================================================
|   TAG |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
|-------+-----------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
| TOTAL |       201 |           3.06 |         614.61 |        259469.020 |       25946.90 | 3 days, 2:57:00 |   186     0    15  92.5 |
===================================================== EXIT REASON STATS =====================================================
|   Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|---------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|           roi |     185 |    185     0     0   100 |           4.32 |         798.39 |        316534     |         798.39 |
|   exit_signal |      15 |      1     0    14   6.7 |         -10.02 |        -150.35 |        -56215.1   |        -150.35 |
|     stop_loss |       1 |      0     0     1     0 |         -33.43 |         -33.43 |          -849.997 |         -33.43 |
======================================================= LEFT OPEN TRADES REPORT ========================================================
|   Pair |   Entries |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|--------+-----------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|  TOTAL |         0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
================== SUMMARY METRICS ==================
| Metric                      | Value               |
|-----------------------------+---------------------|
| Backtesting from            | 2017-08-17 04:00:00 |
| Backtesting to              | 2022-09-30 00:00:00 |
| Max open trades             | 1                   |
|                             |                     |
| Total/Daily Avg Trades      | 201 / 0.11          |
| Starting balance            | 1000 USDT           |
| Final balance               | 260469.02 USDT      |
| Absolute profit             | 259469.02 USDT      |
| Total profit %              | 25946.90%           |
| CAGR %                      | 196.33%             |
| Profit factor               | 5.55                |
| Trades per day              | 0.11                |
| Avg. daily profit %         | 13.88%              |
| Avg. stake amount           | 50861.896 USDT      |
| Total trade volume          | 10223241.176 USDT   |
|                             |                     |
| Best Pair                   | ETH/USDT 614.61%    |
| Worst Pair                  | ETH/USDT 614.61%    |
| Best trade                  | ETH/USDT 13.79%     |
| Worst trade                 | ETH/USDT -33.43%    |
| Best day                    | 27896.094 USDT      |
| Worst day                   | -34204.584 USDT     |
| Days win/draw/lose          | 182 / 1539 / 15     |
| Avg. Duration Winners       | 3 days, 0:57:00     |
| Avg. Duration Loser         | 4 days, 3:44:00     |
| Rejected Entry signals      | 0                   |
| Entry/Exit Timeouts         | 0 / 0               |
|                             |                     |
| Min balance                 | 1000.015 USDT       |
| Max balance                 | 294673.605 USDT     |
| Max % of account underwater | 33.13%              |
| Absolute Drawdown (Account) | 11.61%              |
| Absolute Drawdown           | 34204.584 USDT      |
| Drawdown high               | 293673.605 USDT     |
| Drawdown low                | 259469.02 USDT      |
| Drawdown Start              | 2022-08-13 00:00:00 |
| Drawdown End                | 2022-08-19 10:00:00 |
| Market change               | 340.77%             |
=====================================================
'''

class FastSupertrend_ETHUSDT(IStrategy):
    timeframe = '2h'
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 2.0,
        "buy_m2": 5.0,
        "buy_m3": 16.6,
        "buy_p1": 39,
        "buy_p2": 15,
        "buy_p3": 27,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 1.7,
        "sell_m2": 1.0,
        "sell_m3": 9.9,
        "sell_p1": 36,
        "sell_p2": 12,
        "sell_p3": 33,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.256,
        "448": 0.138,
        "1733": 0.055,
        "4331": 0
    }

    # Stoploss:
    stoploss = -0.333

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.34
    trailing_stop_positive_offset = 0.43
    trailing_only_offset_is_reached = True

    # #--------------- for hyperOpt ---------------
    # minimal_roi = {
    #     "0": 10
    # }
    #
    # # Stoploss:
    # stoploss = -0.99

    buy_m1 = DecimalParameter(1, 20, decimals=1, default=10.0, space='buy')
    buy_m2 = DecimalParameter(1, 20, decimals=1, default=10.0, space='buy')
    buy_m3 = DecimalParameter(1, 20, decimals=1, default=10.0, space='buy')
    buy_p1 = IntParameter(2, 60, default=30, space='buy')
    buy_p2 = IntParameter(2, 60, default=30, space='buy')
    buy_p3 = IntParameter(2, 60, default=30, space='buy')

    sell_m1 = DecimalParameter(1, 20, decimals=1, default=10.0, space='sell')
    sell_m2 = DecimalParameter(1, 20, decimals=1, default=10.0, space='sell')
    sell_m3 = DecimalParameter(1, 20, decimals=1, default=10.0, space='sell')
    sell_p1 = IntParameter(2, 60, default=30, space='sell')
    sell_p2 = IntParameter(2, 60, default=30, space='sell')
    sell_p3 = IntParameter(2, 60, default=30, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # merge 3 buy supertrend to one lines for freqUI
        buy_df_list = [self.supertrend(dataframe, self.buy_m1.value, int(self.buy_p1.value))['ST'],
                       self.supertrend(dataframe, self.buy_m2.value, int(self.buy_p2.value))['ST'],
                       self.supertrend(dataframe, self.buy_m3.value, int(self.buy_p3.value))['ST']]
        buy_df = pd.concat(buy_df_list, keys=range(len(buy_df_list))).groupby(level=1)
        dataframe['supertrend_buy_line'] = buy_df.max()

        # merge 3 sell supertrend to one lines for freqUI
        sell_df_list = [self.supertrend(dataframe, self.sell_m1.value, int(self.sell_p1.value))['ST'],
                        self.supertrend(dataframe, self.sell_m2.value, int(self.sell_p2.value))['ST'],
                        self.supertrend(dataframe, self.sell_m3.value, int(self.sell_p3.value))['ST']]
        sell_df = pd.concat(sell_df_list, keys=range(len(sell_df_list))).groupby(level=1)
        dataframe['supertrend_sell_line'] = sell_df.min()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['supertrend_1_buy'] = self.supertrend(dataframe, self.buy_m1.value, int(self.buy_p1.value))['STX']
        dataframe['supertrend_2_buy'] = self.supertrend(dataframe, self.buy_m2.value, int(self.buy_p2.value))['STX']
        dataframe['supertrend_3_buy'] = self.supertrend(dataframe, self.buy_m3.value, int(self.buy_p3.value))['STX']
        dataframe['supertrend_1_sell'] = self.supertrend(dataframe, self.sell_m1.value, int(self.sell_p1.value))['STX']
        dataframe['supertrend_2_sell'] = self.supertrend(dataframe, self.sell_m2.value, int(self.sell_p2.value))['STX']
        dataframe['supertrend_3_sell'] = self.supertrend(dataframe, self.sell_m3.value, int(self.sell_p3.value))['STX']

        dataframe.loc[
            (
                # The three indicators are 'up' for the current candle
                (dataframe[f'supertrend_1_buy'] == 'up') &
                (dataframe[f'supertrend_2_buy'] == 'up') &
                (dataframe[f'supertrend_3_buy'] == 'up') &
                (dataframe['volume'] > 0)  # There is at least some trading volume
            ), 'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # The three indicators are 'down' for the current candle
                (dataframe[f'supertrend_1_sell'] == 'down') &
                (dataframe[f'supertrend_2_sell'] == 'down') &
                (dataframe[f'supertrend_3_sell'] == 'down') &
                (dataframe['volume'] > 0)  # There is at least some trading volume
            ), 'exit_long'] = 1

        return dataframe

    """
            Supertrend Indicator; adapted for freqtrade
            from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        start_time = time.time()

        df = dataframe.copy()
        last_row = dataframe.tail(1).index.item() + 1

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        UP = ((df['high'] + df['low']) / 2 + multiplier * df['ATR']).values
        DOWN = ((df['high'] + df['low']) / 2 - multiplier * df['ATR']).values

        TRAND_UP = np.zeros(last_row)
        TRAND_DOWN = np.zeros(last_row)
        ST = np.zeros(last_row)
        CLOSE = df['close'].values

        # Compute final upper and lower bands
        for i in range(period, last_row):
            if UP[i] < TRAND_UP[i - 1] or CLOSE[i - 1] > TRAND_UP[i - 1]:
                TRAND_UP[i] = UP[i]
            else:
                TRAND_UP[i] = TRAND_UP[i - 1]

            if DOWN[i] > TRAND_DOWN[i - 1] or CLOSE[i - 1] < TRAND_DOWN[i - 1]:
                TRAND_DOWN[i] = DOWN[i]
            else:
                TRAND_DOWN[i] = TRAND_DOWN[i - 1]

        # Set the Supertrend value
        for i in range(period, last_row):
            if ST[i - 1] == TRAND_UP[i - 1] and CLOSE[i] <= TRAND_UP[i]:
                ST[i] = TRAND_UP[i]
            elif ST[i - 1] == TRAND_UP[i - 1] and CLOSE[i] > TRAND_UP[i]:
                ST[i] = TRAND_DOWN[i]
            elif ST[i - 1] == TRAND_DOWN[i - 1] and CLOSE[i] >= TRAND_DOWN[i]:
                ST[i] = TRAND_DOWN[i]
            elif ST[i - 1] == TRAND_DOWN[i - 1] and CLOSE[i] < TRAND_DOWN[i]:
                ST[i] = TRAND_UP[i]
            else:
                ST[i] = ST[i - 1]

        df_ST = pd.DataFrame(ST, columns=[st])
        df = pd.concat([df, df_ST], axis=1)

        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

        df.fillna(0, inplace=True)

        end_time = time.time()
        # print("total time taken this loop: ", end_time - start_time)

        return DataFrame(index=df.index, data={
            'ST': df[st],
            'STX': df[stx]
        })
