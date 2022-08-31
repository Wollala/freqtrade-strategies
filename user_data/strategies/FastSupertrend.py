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
FastSupertrend_ETHUSDT_20180101-20211231.2022-09-01_080203 이 파일에서 가져옴.
========================================================== BACKTESTING REPORT ==========================================================
|     Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
|----------+--------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
| ETH/USDT |    200 |           2.97 |         593.92 |       2257102.313 |       22571.02 | 3 days, 3:12:00 |   122    63    15  61.0 |
|    TOTAL |    200 |           2.97 |         593.92 |       2257102.313 |       22571.02 | 3 days, 3:12:00 |   122    63    15  61.0 |
========================================================== ENTER TAG STATS ==========================================================
|   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |    Avg Duration |   Win  Draw  Loss  Win% |
|-------+--------+----------------+----------------+-------------------+----------------+-----------------+-------------------------|
| TOTAL |    200 |           2.97 |         593.92 |       2257102.313 |       22571.02 | 3 days, 3:12:00 |   122    63    15  61.0 |
===================================================== EXIT REASON STATS =====================================================
|   Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|---------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|           roi |     184 |    121    63     0   100 |           4.17 |         767.51 |       2.75343e+06 |         767.51 |
|   exit_signal |      16 |      1     0    15   6.2 |         -10.85 |        -173.58 | -496327           |        -173.58 |
====================================================== LEFT OPEN TRADES REPORT ======================================================
|   Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|--------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|  TOTAL |      0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
================== SUMMARY METRICS ==================
| Metric                      | Value               |
|-----------------------------+---------------------|
| Backtesting from            | 2017-08-17 04:00:00 |
| Backtesting to              | 2022-08-19 14:00:00 |
| Max open trades             | 1                   |
|                             |                     |
| Total/Daily Avg Trades      | 200 / 0.11          |
| Starting balance            | 10000 USDT          |
| Final balance               | 2267102.313 USDT    |
| Absolute profit             | 2257102.313 USDT    |
| Total profit %              | 22571.02%           |
| CAGR %                      | 195.34%             |
| Profit factor               | 5.55                |
| Trades per day              | 0.11                |
| Avg. daily profit %         | 12.35%              |
| Avg. stake amount           | 436636.918 USDT     |
| Total trade volume          | 87327383.597 USDT   |
|                             |                     |
| Best Pair                   | ETH/USDT 593.92%    |
| Worst Pair                  | ETH/USDT 593.92%    |
| Best trade                  | ETH/USDT 15.74%     |
| Worst trade                 | ETH/USDT -27.16%    |
| Best day                    | 240401.37 USDT      |
| Worst day                   | -290487.8 USDT      |
| Days win/draw/lose          | 120 / 1601 / 15     |
| Avg. Duration Winners       | 1 day, 16:43:00     |
| Avg. Duration Loser         | 4 days, 3:12:00     |
| Rejected Entry signals      | 0                   |
| Entry/Exit Timeouts         | 0 / 0               |
|                             |                     |
| Min balance                 | 10000 USDT          |
| Max balance                 | 2557590.113 USDT    |
| Max % of account underwater | 29.39%              |
| Absolute Drawdown (Account) | 11.36%              |
| Absolute Drawdown           | 290487.8 USDT       |
| Drawdown high               | 2547590.113 USDT    |
| Drawdown low                | 2257102.313 USDT    |
| Drawdown Start              | 2022-08-12 22:00:00 |
| Drawdown End                | 2022-08-19 10:00:00 |
| Market change               | 460.27%             |
=====================================================
'''


class FastSupertrend(IStrategy):
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
        "0": 0.598,
        "756": 0.136,
        "1711": 0.053,
        "4315": 0
    }

    # Stoploss:
    stoploss = -0.326

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.142
    trailing_stop_positive_offset = 0.146
    trailing_only_offset_is_reached = True

    # --------------- for hyperOpt ---------------
    # minimal_roi = {
    #     "0": 100
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
