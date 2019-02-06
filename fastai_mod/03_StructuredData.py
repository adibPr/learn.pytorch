#!/usr/bin/env python
"""
Classification of structured data using by dividing non-categorical and categorical
data, and use embedding to represent categorical data
"""

# python module
import os
import sys
import glob
import re
from datetime import datetime

# third parties module
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# local module
path_this = os.path.abspath (os.path.dirname (__file__))
sys.path.append (os.path.join (path_this, '..'))

def load_csvs (path) : 
    path_csvs = glob.glob ('{}/*.csv'.format (path))
    csvs = {}
    for p in path_csvs : 
        ds, _ = os.path.splitext (os.path.basename (p))
        csvs[ds] = pd.read_csv (p, low_memory=False)
    return csvs

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: 
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on,
                      suffixes=("", suffix))

def save_samples (datasets, out='./sample.xlsx') : 
    writer = pd.ExcelWriter (out)
    for ds in datasets : 
        datasets[ds][:10].to_excel (writer, ds)

    writer.save ()

def add_datepart (df, date_col, w_drop=False, w_time=False) :
    # so we try to emulate add_datepart from fast.ai
    # the first thing to do is to check wether the coloumn provided is our our our our
    # prefered type
    # we like np.datetime64

    col = df[date_col]
    col_dtype = col.dtype

    if isinstance(col_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        col_dtype = np.datetime64 # because in essence, its just the same

    if not np.issubdtype(col_dtype, np.datetime64): # if not, then convert it
        df[date_col] = col = pd.to_datetime(col, infer_datetime_format=True)

    # we add prefix for our new field name, so we remove all date from col name
    targ_pre = re.sub('[Dd]ate$', '', date_col)
    if targ_pre != '' : 
        targ_pre = targ_pre + '_'

    # all attribute we will add
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    # just in case we deal with time as well
    if w_time: 
        attr = attr + ['Hour', 'Minute', 'Second']

    # turns out the magic is already inside the column, we just call it
    # since the type of the row is np.datetime64, and it has every attribute
    # just call it
    for n in attr: 
        df[targ_pre + n] = getattr(col.dt, n.lower()) # since we can't col.dt.(n.lower ())

    df[targ_pre + 'Elapsed'] = col.astype(np.int64) // 10 ** 9 # dont know the magic

    if w_drop : 
        # drop the original column
        df.drop (date_col, axis=1, inplace=True)

    return df

PATH = '/media/Linux/Learn/pytorch/dataset/rossmann'
datasets = load_csvs (PATH)

"""
for ds in datasets : 
    print ("Dataset : ", ds)
    print (datasets[ds].head ())
    print (datasets[ds].dtypes)
"""


print ('Total data train ', len (datasets['train']))
print ('Total data test ', len (datasets['test']))

# since in train dataset, the is_holiday column type is string, we should convert
# it into boolean
datasets['train']['StateHoliday'] = datasets['train']['StateHoliday'] != '0' # true if x != 0, which means 1. 
datasets['test']['StateHoliday'] = datasets['test']['StateHoliday'] != '0'

# combining information from state names into weather
datasets['weather'] = datasets['weather'].merge (datasets['state_names'], how='left', left_on='file', right_on='StateName')

# splitting google trend week into week_end and week_start, and converting it states
# into formal one
datasets['googletrend']['Date'] = datasets['googletrend']['week'].str.split (' - ', expand=True)[0]
datasets['googletrend']['State'] = datasets['googletrend']['file'].str.split ('_', expand=True)[2]
datasets['googletrend'].loc[datasets['googletrend']['State'] =="NI", "State"] = "HB,NI"

# splitting date into its sub part
for ds in datasets : 
    if 'Date' in datasets[ds].columns : 
        datasets[ds] = add_datepart (datasets[ds], 'Date')

# dont know about this yet
trend_de = datasets['googletrend'][datasets['googletrend']['file'] == 'Rossmann_DE']

# the next thing is to combine everything we have in one datasets
# remember, we have 3 datasets to combine into training and test : 
#   store_states, states, googletrend, weather, and store information

# first, merge store_states into store
datasets['store'] = datasets['store'].merge (datasets['store_states'], how='left', left_on='Store', right_on='Store')
for key in ['train', 'test'] : 
    # merge the merged-store into train/test datasets
    datasets[key] = datasets[key].merge (
            datasets['store'], 
            how='left', 
            left_on='Store', 
            right_on='Store',
            suffixes=("", "_y")
        )

    # merge googletrend based on the same combination State-Year-Week
    datasets[key] = datasets[key].merge (
            datasets['googletrend'], 
            how='left', 
            left_on=['State', 'Year', 'Week'], 
            right_on=['State', 'Year', 'Week'],
            suffixes=("", "_y")
        )

    # merge from trend_de, de stands for ? dont know about this yet
    datasets[key] = datasets[key].merge (
            trend_de, 
            how='left', 
            left_on=['Year', 'Week'], 
            right_on=['Year', 'Week'], 
            suffixes=('', '_DE'),
        )

    # merge the weather
    datasets[key] = datasets[key].merge (
            datasets['weather'],
            how='left',
            left_on=['State', 'Date'],
            right_on=['State', 'Date'],
            suffixes=("", "_y")
        )

    # since if you combined df has same column name, it add affixes _y, we should 
    # remove it since it is a duplicate
    dupl_column = set ([c for c in datasets[key].columns if c.endswith ('_y')])
    for c in dupl_column : 
        datasets[key].drop (c, inplace=True, axis=1)

    # to see which column has null
    # print (datasets[key].isna ().any ())
    # fill na to some columns
    datasets[key]['CompetitionOpenSinceYear'] = datasets[key]['CompetitionOpenSinceYear']\
            .fillna (1900)\
            .astype (np.int32)
    datasets[key]['CompetitionOpenSinceMonth'] = datasets[key]['CompetitionOpenSinceMonth']\
            .fillna (1)\
            .astype (np.int32)
    datasets[key]['Promo2SinceYear'] = datasets[key]['Promo2SinceYear']\
            .fillna (1900)\
            .astype (np.int32)
    datasets[key]['Promo2SinceWeek'] = datasets[key]['Promo2SinceWeek']\
            .fillna (1)\
            .astype (np.int32)

    # create new column, with conversion to datetime of CompetitionOpenSince
    datasets[key]['CompetitionOpenSince'] = pd.to_datetime (
            dict (
                year=datasets[key]['CompetitionOpenSinceYear'],
                month=datasets[key]['CompetitionOpenSinceMonth'],
                day=15 # arbitary
            )
        )
    # column of duration from the store open and the competition open
    datasets[key]['CompetitionDaysOpen'] = datasets[key]['Date'].subtract (
            datasets[key]['CompetitionOpenSince']
        ).dt.days

    # replace errornous/outlying data
    # means that any competitions open first, or it open way back, make it 0
    # to handle outliers in our data
    # df.loc[filter, column] 
    datasets[key].loc[datasets[key]['CompetitionDaysOpen'] < 0, 'CompetitionDaysOpen'] = 0
    datasets[key].loc[datasets[key]['CompetitionOpenSinceYear'] < 1990, 'CompetitionDaysOpen'] = 0

    # new column, duration (in month) of competitor
    datasets[key]['CompetitionMonthsOpen'] = datasets[key]['CompetitionDaysOpen'] // 30 
    # if we found competitions with difference larger than 2 years, make it just 2 years
    # to limited our disparcy data
    datasets[key].loc[datasets[key]['CompetitionMonthsOpen'] > 24, 'CompetitionMonthsOpen'] = 24

    # the same apply into Promo
    # datasets[key]['Promo2Since'] = pd.to_datetime (datasets[key].apply (
    #         lambda row : datetime.strptime ('{}-W{}-0'.format (row['Promo2SinceYear'], row['Promo2SinceWeek']), '%Y-W%W-%w'), 
    #         axis=1
    #     ).astype (pd.datetime))

# save_samples (datasets)
