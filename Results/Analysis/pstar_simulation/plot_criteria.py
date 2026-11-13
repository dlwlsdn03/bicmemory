import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ps,wd,vol=pd.read_pickle('criteria_results.pkl')
INK,MUTED,GRID='#1f1f1e','#6b6a65','#e4e3df'
COL={'AIC':'#2a78d6','HQC':'#1baf7a','BIC':'#eb6834'}
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
  'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6})
fig,axs=plt.subplots(1,2,figsize=(11,4.2),sharey=True,gridspec_kw=dict(wspace=0.08))
for ax,(w,title) in zip(axs,[('expanding','A. Expanding window (main spec)'),('rolling5y','B. Rolling 5-year window')]):
    ax.axhspan(0,10,color=MUTED,alpha=0.07,lw=0)
    for c in ['AIC','HQC','BIC']:
        q=ps[(c,w)].quantile([.25,.5,.75],axis=1).T
        ax.fill_between(q.index,q[.25],q[.75],color=COL[c],alpha=0.14,lw=0)
        ax.plot(q.index,q[.5],color=COL[c],lw=2,label=f'{c} median (band = IQR)')
    ax.set_title(title,loc='left',fontsize=10,color=INK); ax.set_ylim(0,45)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4)); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
axs[0].set_ylabel('Selected lookback $p^*$ (27 stocks)')
axs[0].legend(frameon=False,loc='upper left',fontsize=8)
axs[0].text(pd.Timestamp('2004-03-01'),1,'Grey band: below the shortest fixed lookback (10)',color=MUTED,fontsize=7.5)
fig.text(0.125,-0.04,'Monthly re-selection, AR(p) with intercept on log QMLE-Trade RV, p = 1..63, k = p + 2. Penalties per parameter: AIC 2, HQC 2 log log n, BIC log n.',fontsize=7.5,color=MUTED)
fig.savefig('pstar_by_criterion.png',dpi=180,bbox_inches='tight')
