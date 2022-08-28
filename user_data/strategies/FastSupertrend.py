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
FastSupertrend_ETHUSDT_20180101-20211231.2022-08-29_020206 이 파일에서 가져옴.
실제론 2h 결과 2개가 있는데 (아래 참고)
1. |   Best | 1868/2000 |      162 |     99   47   16 |        2.66% |   479238.821 USDT (4,792.39%) | 3 days 00:24:00 | -479,238.82121 |     89620.801 USDT   (21.52%) |
2. |   Best | 1490/2000 |      175 |    104   57   14 |        2.51% |   499028.520 USDT (4,990.29%) | 3 days 04:04:00 | -494,052.92164 |     64725.023 USDT   (13.03%) |
최근 벡테스팅(2022.01.01~2022.08.20) 과 전체 기간 백테스팅(2017.01.01~2022.08.20)은 1번이 더 좋아서 1번을 사용하기로 함. 아래가 1번의 결과임.

=========================================================== BACKTESTING REPORT ==========================================================
|     Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
|----------+--------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
| ETH/USDT |    179 |           2.59 |         464.09 |         64051.268 |        6405.13 | 2 days, 23:30:00 |   109    52    18  60.9 |
|    TOTAL |    179 |           2.59 |         464.09 |         64051.268 |        6405.13 | 2 days, 23:30:00 |   109    52    18  60.9 |
========================================================== ENTER TAG STATS ===========================================================
|   TAG |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |     Avg Duration |   Win  Draw  Loss  Win% |
|-------+--------+----------------+----------------+-------------------+----------------+------------------+-------------------------|
| TOTAL |    179 |           2.59 |         464.09 |         64051.268 |        6405.13 | 2 days, 23:30:00 |   109    52    18  60.9 |
===================================================== EXIT REASON STATS =====================================================
|   Exit Reason |   Exits |   Win  Draws  Loss  Win% |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |
|---------------+---------+--------------------------+----------------+----------------+-------------------+----------------|
|           roi |     161 |    109    52     0   100 |           4.19 |         675.09 |          108998   |         675.09 |
|   exit_signal |      18 |      0     0    18     0 |         -11.72 |        -211    |          -44947.1 |        -211    |
====================================================== LEFT OPEN TRADES REPORT ======================================================
|   Pair |   Buys |   Avg Profit % |   Cum Profit % |   Tot Profit USDT |   Tot Profit % |   Avg Duration |   Win  Draw  Loss  Win% |
|--------+--------+----------------+----------------+-------------------+----------------+----------------+-------------------------|
|  TOTAL |      0 |           0.00 |           0.00 |             0.000 |           0.00 |           0:00 |     0     0     0     0 |
================== SUMMARY METRICS ==================
| Metric                      | Value               |
|-----------------------------+---------------------|
| Backtesting from            | 2018-01-01 00:00:00 |
| Backtesting to              | 2022-08-19 14:00:00 |
| Max open trades             | 1                   |
|                             |                     |
| Total/Daily Avg Trades      | 179 / 0.11          |
| Starting balance            | 1000 USDT           |
| Final balance               | 65051.268 USDT      |
| Absolute profit             | 64051.268 USDT      |
| Total profit %              | 6405.13%            |
| CAGR %                      | 146.26%             |
| Profit factor               | 2.43                |
| Trades per day              | 0.11                |
| Avg. daily profit %         | 3.79%               |
| Avg. stake amount           | 18911.94 USDT       |
| Total trade volume          | 3385237.331 USDT    |
|                             |                     |
| Best Pair                   | ETH/USDT 464.09%    |
| Worst Pair                  | ETH/USDT 464.09%    |
| Best trade                  | ETH/USDT 17.95%     |
| Worst trade                 | ETH/USDT -21.71%    |
| Best day                    | 7300.419 USDT       |
| Worst day                   | -9645.048 USDT      |
| Days win/draw/lose          | 108 / 1462 / 18     |
| Avg. Duration Winners       | 1 day, 22:44:00     |
| Avg. Duration Loser         | 4 days, 15:53:00    |
| Rejected Entry signals      | 0                   |
| Entry/Exit Timeouts         | 0 / 0               |
|                             |                     |
| Min balance                 | 1033.234 USDT       |
| Max balance                 | 74696.316 USDT      |
| Max % of account underwater | 31.61%              |
| Absolute Drawdown (Account) | 12.91%              |
| Absolute Drawdown           | 9645.048 USDT       |
| Drawdown high               | 73696.316 USDT      |
| Drawdown low                | 64051.268 USDT      |
| Drawdown Start              | 2022-08-13 08:00:00 |
| Drawdown End                | 2022-08-19 08:00:00 |
| Market change               | 136.52%             |
=====================================================
'''


class FastSupertrend(IStrategy):
    timeframe = '2h'
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 1.9,
        "buy_m2": 5.3,
        "buy_m3": 15.4,
        "buy_p1": 17,
        "buy_p2": 22,
        "buy_p3": 46,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 4.9,
        "sell_m2": 9.0,
        "sell_m3": 2.4,
        "sell_p1": 23,
        "sell_p2": 21,
        "sell_p3": 60,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.292,
        "762": 0.135,
        "1632": 0.087,
        "3480": 0
    }

    # Stoploss:
    stoploss = -0.272

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

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

