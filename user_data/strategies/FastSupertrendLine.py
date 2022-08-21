"""
Supertrend strategy:
* Description: Generate a 3 supertrend indicators for 'buy' strategies & 3 supertrend indicators for 'sell' strategies
               Buys if the 3 'buy' indicators are 'up'
               Sells if the 3 'sell' indicators are 'down'
* Author: @juankysoriano (Juan Carlos Soriano)
* github: https://github.com/juankysoriano/

*** NOTE: This Supertrend strategy is just one of many possible strategies using `Supertrend` as indicator. It should on any case used at your own risk.
          It comes with at least a couple of caveats:
            1. The implementation for the `supertrend` indicator is based on the following discussion: https://github.com/freqtrade/freqtrade-strategies/issues/30 . Concretelly https://github.com/freqtrade/freqtrade-strategies/issues/30#issuecomment-853042401
            2. The implementation for `supertrend` on this strategy is not validated; meaning this that is not proven to match the results by the paper where it was originally introduced or any other trusted academic resources
"""

import time

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter


class FastSupertrendLine(IStrategy):
    # minimal_roi = {
    #     "0": 10
    # }
    # stoploss = -0.99
    timeframe = '2h'
    # Buy hyperspace params:
    buy_params = {
        "buy_m1": 11.0,
        "buy_m2": 14.8,
        "buy_m3": 4.5,
        "buy_p1": 13,
        "buy_p2": 58,
        "buy_p3": 13,
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_m1": 11.9,
        "sell_m2": 6.9,
        "sell_m3": 7.6,
        "sell_p1": 26,
        "sell_p2": 47,
        "sell_p3": 40,
    }

    # ROI table:
    minimal_roi = {
        "0": 0.622,
        "387": 0.138,
        "1408": 0.081,
        "3600": 0
    }

    # Stoploss:
    stoploss = -0.348

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    buy_m1 = DecimalParameter(1, 15, decimals=1, default=7.1, space='buy')
    buy_m2 = DecimalParameter(1, 15, decimals=1, default=7.1, space='buy')
    buy_m3 = DecimalParameter(1, 15, decimals=1, default=7.1, space='buy')
    buy_p1 = IntParameter(2, 60, default=30, space='buy')
    buy_p2 = IntParameter(2, 60, default=30, space='buy')
    buy_p3 = IntParameter(2, 60, default=30, space='buy')

    sell_m1 = DecimalParameter(1, 15, decimals=1, default=7.1, space='sell')
    sell_m2 = DecimalParameter(1, 15, decimals=1, default=7.1, space='sell')
    sell_m3 = DecimalParameter(1, 15, decimals=1, default=7.1, space='sell')
    sell_p1 = IntParameter(2, 60, default=30, space='sell')
    sell_p2 = IntParameter(2, 60, default=30, space='sell')
    sell_p3 = IntParameter(2, 60, default=30, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['supertrend_1_buy'] = self.supertrend(dataframe, self.buy_m1.value, int(self.buy_p1.value))['ST']
        dataframe['supertrend_2_buy'] = self.supertrend(dataframe, self.buy_m2.value, int(self.buy_p2.value))['ST']
        dataframe['supertrend_3_buy'] = self.supertrend(dataframe, self.buy_m3.value, int(self.buy_p3.value))['ST']
        dataframe['supertrend_1_sell'] = self.supertrend(dataframe, self.sell_m1.value, int(self.sell_p1.value))['ST']
        dataframe['supertrend_2_sell'] = self.supertrend(dataframe, self.sell_m2.value, int(self.sell_p2.value))['ST']
        dataframe['supertrend_3_sell'] = self.supertrend(dataframe, self.sell_m3.value, int(self.sell_p3.value))['ST']

        dataframe.loc[
            (
                    (dataframe[f'supertrend_1_buy'] < dataframe['close']) &
                    (dataframe[f'supertrend_2_buy'] < dataframe['close']) &
                    (dataframe[f'supertrend_3_buy'] < dataframe['close']) &  # The three indicators are 'up' for the current candle
                    (dataframe['volume'] > 0)  # There is at least some trading volume
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe[f'supertrend_1_sell'] > dataframe['close']) &
                    (dataframe[f'supertrend_2_sell'] > dataframe['close']) &
                    (dataframe[f'supertrend_3_sell'] > dataframe['close']) &  # The three indicators are 'down' for the current candle
                    (dataframe['volume'] > 0)  # There is at least some trading volume
            ),
            'exit_long'] = 1

        return dataframe

    """
        Supertrend Indicator; adapted for freqtrade
        from: https://github.com/freqtrade/freqtrade-strategies/issues/30
    """

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        start_time = time.time()

        df = dataframe.copy()
        last_row = dataframe.tail(1).index.item()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        BASIC_UB = ((df['high'] + df['low']) / 2 + multiplier * df['ATR']).values
        BASIC_LB = ((df['high'] + df['low']) / 2 - multiplier * df['ATR']).values

        FINAL_UB = np.zeros(last_row + 1)
        FINAL_LB = np.zeros(last_row + 1)
        ST = np.zeros(last_row + 1)
        CLOSE = df['close'].values

        # Compute final upper and lower bands
        for i in range(period, last_row):
            FINAL_UB[i] = BASIC_UB[i] if BASIC_UB[i] < FINAL_UB[i - 1] or CLOSE[i - 1] > FINAL_UB[i - 1] else FINAL_UB[
                i - 1]
            FINAL_LB[i] = BASIC_LB[i] if BASIC_LB[i] > FINAL_LB[i - 1] or CLOSE[i - 1] < FINAL_LB[i - 1] else FINAL_LB[
                i - 1]

        # Set the Supertrend value
        for i in range(period, last_row):
            ST[i] = FINAL_UB[i] if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] <= FINAL_UB[i] else \
                FINAL_LB[i] if ST[i - 1] == FINAL_UB[i - 1] and CLOSE[i] > FINAL_UB[i] else \
                    FINAL_LB[i] if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] >= FINAL_LB[i] else \
                        FINAL_UB[i] if ST[i - 1] == FINAL_LB[i - 1] and CLOSE[i] < FINAL_LB[i] else 0.00
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
