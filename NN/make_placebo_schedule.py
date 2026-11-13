"""Build the placebo lookback schedule for LSTM_PLACEBO.

For every January re-selection date (2004-2026) this computes each stock's
AIC-selected lookback exactly as LSTM_AIC does in NN.ipynb (QMLE-Trade log RV,
duplicate symbol-dates dropped, selection sample = 1 Jan 2000 up to the
selection date, AR(p) with intercept, p = 1..63, common hold-back of 63), then
writes the cross-stock median as one lookback shared by all stocks.

Only data before each selection date are used, so the schedule has no
look-ahead. Nested AR fits come from one Cholesky factor per stock and date;
this reproduces statsmodels AutoReg(..., hold_back=63).aic selections
(checked with --verify).

Usage (from NN/):  python make_placebo_schedule.py [--verify]
Writes: lookback_schedule_placebo.csv  (selection_date, lookback, n_stocks)
        aic_lookbacks_by_stock.csv     (selection_date x ticker, for reference)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'Data')
P_MAX = 63
SAMPLE_START = pd.Timestamp('2000-01-01')
SAMPLE_END = pd.Timestamp('2026-06-30')
FIRST_SELECTION = pd.Timestamp('2004-01-01')
RESELECT_MONTHS = 12


def load_log_rv(path):
    frame = pd.read_csv(path)
    frame = frame[frame['Type'].eq('QMLE-Trade')][['Date', 'Volatility']].copy()
    text = frame['Date'].astype(str).str.strip()
    slash = text.str.contains('/', regex=False)
    dates = pd.Series(pd.NaT, index=frame.index, dtype='datetime64[ns]')
    dates[~slash] = pd.to_datetime(text[~slash], format='%Y-%m-%d')
    dates[slash] = pd.to_datetime(text[slash], dayfirst=True)
    frame['Date'] = dates
    frame = frame[~frame['Date'].duplicated(keep=False)]      # duplicate_policy='drop'
    series = frame.set_index('Date')['Volatility'].sort_index()
    series = series[(series.index >= SAMPLE_START) & (series.index <= SAMPLE_END)]
    return np.log(series.clip(lower=1e-8))


def aic_order(values):
    """AIC-selected AR order with a common hold-back of P_MAX observations."""
    t = len(values)
    design = np.column_stack(
        [np.ones(t - P_MAX)]
        + [values[P_MAX - k:t - k] for k in range(1, P_MAX + 1)]
        + [values[P_MAX:]]
    )
    n = t - P_MAX
    upper = np.linalg.cholesky(design.T @ design).T
    tail = np.cumsum((upper[:, -1] ** 2)[::-1])[::-1]
    rss = tail[2:P_MAX + 2]                       # orders 1..P_MAX
    orders = np.arange(1, P_MAX + 1)
    aic = n * np.log(rss / n) + 2 * (orders + 2)
    return int(np.argmin(aic)) + 1


def main(verify=False):
    dates = pd.date_range(FIRST_SELECTION, SAMPLE_END, freq=f'{RESELECT_MONTHS}MS')
    by_stock = {}
    for path in sorted(glob.glob(os.path.join(DATA, '*.csv'))):
        ticker = os.path.basename(path)[:-4]
        log_rv = load_log_rv(path)
        by_stock[ticker] = [aic_order(log_rv[log_rv.index < d].values) for d in dates]
    by_stock = pd.DataFrame(by_stock, index=dates)
    by_stock.index.name = 'selection_date'

    median = by_stock.median(axis=1)
    if not np.allclose(median, median.round()):
        raise ValueError('Cross-stock median is not an integer; use an odd number of stocks')
    schedule = pd.DataFrame({
        'selection_date': dates.date,
        'lookback': median.round().astype(int).values,
        'n_stocks': by_stock.shape[1],
    })
    schedule.to_csv(os.path.join(HERE, 'lookback_schedule_placebo.csv'), index=False)
    by_stock.to_csv(os.path.join(HERE, 'aic_lookbacks_by_stock.csv'))
    print(schedule.to_string(index=False))

    if verify:
        from statsmodels.tsa.ar_model import AutoReg
        checks = [('CSCO', '2008-01-01'), ('HON', '2015-01-01'), ('KO', '2021-01-01')]
        for ticker, date in checks:
            log_rv = load_log_rv(os.path.join(DATA, f'{ticker}.csv'))
            sample = log_rv[log_rv.index < pd.Timestamp(date)].reset_index(drop=True)
            aic = {p: AutoReg(sample, lags=p, hold_back=P_MAX).fit().aic
                   for p in range(1, P_MAX + 1)}
            expected = min(aic, key=aic.get)
            got = by_stock.loc[pd.Timestamp(date), ticker]
            print(f'verify {ticker} {date}: AutoReg {expected}, here {got}')
            if expected != got:
                raise AssertionError('Mismatch with statsmodels AutoReg')


if __name__ == '__main__':
    main(verify='--verify' in sys.argv)
