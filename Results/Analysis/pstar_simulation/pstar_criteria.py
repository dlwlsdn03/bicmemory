"""p* under AIC, HQC and BIC, for expanding and rolling (3/5/10y) selection windows.

Same AR(p) fits as pstar_simulation.py; criteria use k = p + 2 and
n = observations after the common hold-back of 63:
  AIC = n log(RSS/n) + 2k,  HQC = ... + 2k log log n,  BIC = ... + k log n
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pstar_simulation import load, DATA, OUT, P, START, END, FIRST_TEST

PEN = {'AIC': lambda n: 2.0, 'HQC': lambda n: 2.0 * np.log(np.log(n)), 'BIC': lambda n: np.log(n)}
WINDOWS = {'expanding': None, 'rolling10y': 10, 'rolling5y': 5, 'rolling3y': 3}

def loglik_profile(v):
    T = len(v)
    Z = np.column_stack([np.ones(T - P)] + [v[P - k:T - k] for k in range(1, P + 1)] + [v[P:]])
    n = T - P
    R = np.linalg.cholesky(Z.T @ Z).T
    tail = np.cumsum((R[:, -1] ** 2)[::-1])[::-1]
    return n, n * np.log(tail[2:P + 2] / n)

def run():
    months = pd.date_range(FIRST_TEST, END, freq='MS')
    k = np.arange(1, P + 1) + 2
    out = {(c, w): {} for c in PEN for w in WINDOWS}
    width = {(c, w): {} for c in PEN for w in WINDOWS}
    vol = {}
    for f in sorted(glob.glob(os.path.join(DATA, '*.csv'))):
        tic = os.path.basename(f)[:-4]
        y = load(f); rv = np.exp(y)
        vol[tic] = [rv[y.index < ts].iloc[-22:].mean() for ts in months]
        for w, yrs in WINDOWS.items():
            res = {c: [] for c in PEN}; wd = {c: [] for c in PEN}
            for ts in months:
                st = START if yrs is None else max(START, ts - pd.DateOffset(years=yrs))
                n, ll = loglik_profile(y.values[(y.index >= st) & (y.index < ts)])
                for c, pen in PEN.items():
                    ic = ll + pen(n) * k
                    res[c].append(int(np.argmin(ic)) + 1)
                    ok = np.where(ic - ic.min() < 2)[0] + 1; wd[c].append(ok.max() - ok.min())
            for c in PEN:
                out[(c, w)][tic] = res[c]; width[(c, w)][tic] = wd[c]
    mk = lambda d: pd.DataFrame(d, index=months)
    return {key: mk(v) for key, v in out.items()}, {key: mk(v) for key, v in width.items()}, mk(vol)

if __name__ == '__main__':
    ps, wd, vol = run()
    for (c, w), df in ps.items():
        df.to_csv(os.path.join(OUT, f'pstar_monthly_{c}_{w}.csv'), index_label='test_start')
    pd.to_pickle((ps, wd, vol), os.path.join(OUT, 'criteria_results.pkl'))
    print('done')
