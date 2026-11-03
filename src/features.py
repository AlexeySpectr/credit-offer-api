import numpy as np
import pandas as pd


def build_features(df):
    data = df.copy()

    date = pd.to_datetime(
        data['decision_day'],
        errors='coerce'
    )

    
    data['weekends'] = date.dt.dayofweek.isin([5, 6]).astype(int)

    month = date.dt.month
    week = date.dt.isocalendar().week.astype(float)

    data['month_sin'] = np.sin(2 * np.pi * month / 12)
    data['month_cos'] = np.cos(2 * np.pi * month / 12)

    data['week_sin'] = np.sin(2 * np.pi * week / 52)
    data['week_cos'] = np.cos(2 * np.pi * week / 52)

    
    data['rate_spread'] = (
        data['offered_rate'] - data['cb_rate']
    )

    
    data['is_fl_miss_0_or_1'] = (
        data['fl_adminarea'].isna().astype(int)
    )

    data['fl_adminarea'] = (
        data['fl_adminarea'].fillna('on_phone')
    )

    
    feature_cols = [
        'loan_amount_last',
        'overdraft_limit_min',
        'overdraft_limit_max',
        'offered_rate',
        'cb_rate',
        'rate_spread',
        'weekends',
        'month_sin',
        'month_cos',
        'week_sin',
        'week_cos',
        'fl_adminarea',
        'is_fl_miss_0_or_1'
    ]

    return data[feature_cols]