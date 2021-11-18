#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:                  gen_market_data.py
usage:                  python gen_market_data.py
python version:         3.9

author:                 Raja Grewal
email:                  raja_grewal1@pm.me
website:                https://github.com/rgrewa1

Description:
    Collects market data from online sources and creates NumPy arrays ready for 
    agent learning. All data is remotely (and freely) obtained using pandas-datareader 
    following https://pydata.github.io/pandas-datareader/remote_data.html.

    Historical data for major indices, commodities, and currencies is obtained from 
    Stooq at https://stooq.com/. Note not every symbol can be utilised, all must be
    individually checked to determine feasibility. 

Instructions:
    1. Select appropriate start and end date for data for all assets with daily data 
       sampling frequency.
    2. Enter into dictionary the obtained Stooq symbols for desired assets and place in list
       following the naming scheme.
    3. Running file will scrape data and place it in a directory containing pickle 
       and csv files, along with a cleaned NumPy array.

Stooq - Symbols and Data Availability:
    ^SPX: S&P 500                       https://stooq.com/q/d/?s=^spx
    ^DJI: Dow Jones Industrial 30 Cash  https://stooq.com/q/d/?s=^dji
    ^NDX: Nasdaq 100                    https://stooq.com/q/d/?s=^ndx

    GC.F: Gold - COMEX                  https://stooq.com/q/d/?s=gc.f
    SI.F: Silver - COMEX                https://stooq.com/q/d/?s=si.f
    HG.F: High Grade Copper - COMEX     https://stooq.com/q/d/?s=hg.f
    PL.F: Platinum - NYMEX              https://stooq.com/q/d/?s=pf.f
    PA.F: Palladium - NYMEX             https://stooq.com/q/d/?s=pa.f

    CL.F: Crude Oil WTI- NYMEX          https://stooq.com/q/d/?s=cl.f
    RB.F: Gasoline RBOB - NYMEX         https://stooq.com/q/d/?s=rb.f

    LS.F: Lumber - CME                  https://stooq.com/q/d/?s=ls.f
    LE.F: Live Cattle - CME             https://stooq.com/q/d/?s=le.f
    KC.F: Coffee - ICE                  https://stooq.com/q/d/?s=kc.f
    OJ.F: Orange Juice - ICE            https://stooq.com/q/d/?s=oj.f

    ^ = index value
    .C = cash
    .F = front month futures
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import os

# common starting/endiing dates for daily data collection for all assets
start: str = '1985-10-01'
end: str = '2021-11-12'

# pairs for data saving and assets to be included
stooq: dict = {
    'name0': 'snp',
    'assets0': ['^SPX'],

    'name1': 'usei',
    'assets1': ['^SPX', '^DJI', '^NDX'],
    
    'name2': 'minor',
    'assets2': ['^SPX', '^DJI', '^NDX', 
                'GC.F', 'SI.F',  
                'CL.F'],

    'name3': 'medium',
    'assets3': ['^SPX', '^DJI', '^NDX', 
                'GC.F', 'SI.F', 'HG.F', 'PL.F',
                'CL.F', 
                'LS.F'],

    'name4': 'major',
    'assets4': ['^SPX', '^DJI', '^NDX', 
                'GC.F', 'SI.F', 'HG.F', 'PL.F', 'PA.F', 
                'CL.F', 'RB.F', 
                'LS.F', 'LE.F', 'KC.F', 'OJ.F'],

    'name5': 'dji',
    'assets5': ['^SPX', '^DJI', '^NDX', 
                'AAPL.US', 'AXP.US', 'BA.US', 'CAT.US', 'CVX.US', 'DIS.US',  
                'HD.US', 'IBM.US', 'INTC.US', 'JNJ.US', 'JPM.US', 'KO.US', 
                'MCD.US', 'MMM.US', 'MRK.US', 'MSFT.US', 'NKE.US', 'PFE.US', 
                'PG.US', 'RTX.US', 'VZ.US', 'WBA.US', 'WMT.US', 'XOM.US',
                'CSCO.US', 'UNH.US',                     # starts 1990
                # 'DOW.US', 'GS.US', 'TRV.US', 'V.US'    # very little data
                ],

    'name6': 'full',
    'assets6': ['^SPX', '^DJI', '^NDX', 
                'GC.F', 'SI.F', 'HG.F', 'PL.F', 'PA.F', 
                'CL.F', 'RB.F', 
                'LS.F', 'LE.F', 'KC.F', 'OJ.F',
                'AAPL.US', 'AXP.US', 'BA.US', 'CAT.US', 'CVX.US', 'DIS.US',  
                'HD.US', 'IBM.US', 'INTC.US', 'JNJ.US', 'JPM.US', 'KO.US', 
                'MCD.US', 'MMM.US', 'MRK.US', 'MSFT.US', 'NKE.US', 'PFE.US', 
                'PG.US', 'RTX.US', 'VZ.US', 'WBA.US', 'WMT.US', 'XOM.US',
                'CSCO.US', 'UNH.US',                     # starts 1990
                # 'DOW.US', 'GS.US', 'TRV.US', 'V.US'    # very little data
                ],
    }

def dataframe_to_array(market_data: pd.DataFrame, price_type: str) -> np.ndarray:
    """
    Converts pandas dataframe to cleaned numpy array by extracting relevant prices.

    Parameters:
        markert_data: raw dataframe generated by pandas_datareader from remote source
        price_type: 'Open', 'High', 'Low', or 'Close' prices for the time step

    Returns:
        prices: array of asset prices of a given type
    """
    market = market_data[str(price_type)]

    # remove all rows with missing values
    market = market.dropna(how='any')

    # format time ordering if needed (earliest data point is at index 0)
    if market.index[0] > market.index[-1]:
        market = market[::-1]

    n_assets, n_days = market.columns.shape[0], market.index.shape[0]

    prices = np.empty((n_days, n_assets), dtype=np.float64)

    a = 0
    for asset in market.columns:
        prices[:, a] = market[str(asset)]
        a += 1

    return prices

if __name__ == '__main__':

    dir = './docs/market_data/'    # directory for saving market prices dataframes, csvs, and arrays

    if not os.path.exists(dir):
        os.makedirs(dir)

    for x in range(0, int(len(stooq) / 2)):
        name = 'stooq_' + stooq['name'+str(x)]
        assets = stooq['assets'+str(x)]

        scraped_data = pdr.get_data_stooq(assets, start, end)
        scraped_data.to_pickle(dir + name + '.pkl')

        market = pd.read_pickle(dir + name + '.pkl')

        market.to_csv(dir + name + '.csv')
        
        prices = dataframe_to_array(market, 'Close')
        np.save(dir + name + '.npy', prices)

        print('{}: n_assets = {}, days = {}'.format(name, prices.shape[1], prices.shape[0]))