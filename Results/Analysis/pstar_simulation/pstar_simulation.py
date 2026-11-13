"""Simulate the AIC-selected lookback p* under monthly vs annual re-selection.

Mirrors NN.ipynb: QMLE-Trade log RV, duplicate symbol-dates dropped, selection
sample = [2000-01-01, test_start), AR(p) with intercept by conditional ML,
hold_back = 63, p in 1..63. Nested OLS fits are computed from one Cholesky
factor of the augmented cross-product matrix per window.
"""
import glob, os, sys
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'Data')
OUT = os.path.dirname(os.path.abspath(__file__))
P = 63
START, END = pd.Timestamp('2000-01-01'), pd.Timestamp('2026-06-30')
FIRST_TEST = pd.Timestamp('2004-01-01')

def load(path):
    d = pd.read_csv(path)
    d = d[d.Type.eq('QMLE-Trade')][['Date', 'Volatility']].copy()
    txt = d.Date.astype(str).str.strip()
    sl = txt.str.contains('/', regex=False)
    dt = pd.Series(pd.NaT, index=d.index, dtype='datetime64[ns]')
    dt[~sl] = pd.to_datetime(txt[~sl], format='%Y-%m-%d')
    dt[sl] = pd.to_datetime(txt[sl], dayfirst=True)
    d['Date'] = dt
    d = d[~d.Date.duplicated(keep=False)]          # duplicate_policy='drop'
    s = d.set_index('Date').Volatility.sort_index()
    s = s[(s.index >= START) & (s.index <= END)]
    return np.log(s.clip(lower=1e-8))

def selection_path(y, test_starts):
    """p* and the AIC profile at each test start (expanding sample from START)."""
    v, idx = y.values, y.index
    T = len(v)
    # rows t = P..T-1: [1, y_{t-1}, ..., y_{t-P}, y_t]
    Z = np.column_stack([np.ones(T - P)] + [v[P - k:T - k] for k in range(1, P + 1)] + [v[P:]])
    rdates = idx[P:]
    A = np.zeros((P + 2, P + 2)); used = 0; out = []
    for ts in test_starts:
        n = int(np.searchsorted(rdates.values, np.datetime64(ts)))  # rows with date < ts
        if n > used:
            A += Z[used:n].T @ Z[used:n]; used = n
        R = np.linalg.cholesky(A).T                     # A = R'R, R upper
        r = R[:, -1] ** 2
        tail = np.cumsum(r[::-1])[::-1]                 # tail[j] = sum_{i>=j} r_i
        rss = tail[2:P + 2]                             # p = 1..P  -> first p+1 regressors
        p = np.arange(1, P + 1)
        aic = n * np.log(rss / n) + 2 * (p + 2)
        out.append((n, aic))
    return out

def main():
    files = sorted(glob.glob(os.path.join(DATA, '*.csv')))
    months = pd.date_range(FIRST_TEST, END, freq='MS')
    years = pd.date_range(FIRST_TEST, END, freq='YS')
    mon, ann, prof_m = {}, {}, {}
    for f in files:
        tic = os.path.basename(f)[:-4]
        y = load(f)
        res_m = selection_path(y, months)
        res_a = selection_path(y, years)
        mon[tic] = [int(np.argmin(a)) + 1 for _, a in res_m]
        ann[tic] = [int(np.argmin(a)) + 1 for _, a in res_a]
        prof_m[tic] = np.vstack([a for _, a in res_m])
    pm = pd.DataFrame(mon, index=months); pa = pd.DataFrame(ann, index=years)
    pm.to_csv(os.path.join(OUT, 'pstar_monthly.csv'), index_label='test_start')
    pa.to_csv(os.path.join(OUT, 'pstar_annual.csv'), index_label='test_start')
    np.savez_compressed(os.path.join(OUT, 'aic_profiles_monthly.npz'), **prof_m)
    print('done', pm.shape, pa.shape)

if __name__ == '__main__':
    main()
