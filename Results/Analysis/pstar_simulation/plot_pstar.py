import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pm=pd.read_csv('pstar_monthly.csv',index_col=0,parse_dates=True)
pa=pd.read_csv('pstar_annual.csv',index_col=0,parse_dates=True)
BLUE,ORANGE,INK,MUTED,GRID='#2a78d6','#eb6834','#1f1f1e','#6b6a65','#e4e3df'
plt.rcParams.update({'font.size':9,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':0.6})
end=pd.Timestamp('2026-07-01')
def step(df):  # extend annual series to the end so the last step is drawn
    s=df.copy(); s.loc[end]=s.iloc[-1]; return s
fig=plt.figure(figsize=(10,7.2))
gs=fig.add_gridspec(3,4,height_ratios=[1.25,1,1],hspace=0.75,wspace=0.28)
ax=fig.add_subplot(gs[0,:])
q=pm.quantile([.25,.5,.75],axis=1).T
ax.fill_between(q.index,q[.25],q[.75],color=BLUE,alpha=0.15,lw=0,label='Monthly IQR')
ax.plot(q.index,q[.5],color=BLUE,lw=2,label='Monthly median')
qa=step(pa.quantile([.25,.5,.75],axis=1).T)
ax.step(qa.index,qa[.5],where='post',color=ORANGE,lw=2,label='Annual median')
ax.step(qa.index,qa[.25],where='post',color=ORANGE,lw=1,ls='--',label='Annual IQR')
ax.step(qa.index,qa[.75],where='post',color=ORANGE,lw=1,ls='--')
ax.axhline(63,color=MUTED,lw=1,ls=':'); ax.text(pd.Timestamp('2026-06-30'),64.5,'cap = 63',color=MUTED,fontsize=8,ha='right',va='bottom')
ax.set_ylim(0,70); ax.set_ylabel('AIC-selected lookback $p^*$')
ax.set_title('A. Cross-section of 27 stocks: monthly vs annual re-selection',loc='left',color=INK,fontsize=10)
ax.legend(ncol=4,frameon=False,loc='upper left',bbox_to_anchor=(0,-0.12),fontsize=8)
axs=[]
for k,t in enumerate(['CSCO','HON','UNH','TRV','HD','KO','MMM','CAT']):
    a=fig.add_subplot(gs[1+k//4,k%4])
    a.plot(pm.index,pm[t],color=BLUE,lw=1.4)
    sa=step(pa[[t]])[t]; a.step(sa.index,sa,where='post',color=ORANGE,lw=1.6)
    a.set_ylim(0,66); a.set_yticks([0,21,42,63]); a.set_title(t,loc='left',fontsize=9,color=INK)
    a.xaxis.set_major_locator(matplotlib.dates.YearLocator(8)); a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    if k%4: a.set_yticklabels([])
    axs.append(a)
p=axs[0].get_position(); fig.text(p.x0,p.y1+0.045,'B. Example stocks (blue = monthly, orange = annual)',fontsize=10,color=INK)
fig.text(0.125,0.02,'Expanding selection sample from 1 Jan 2000; AR(p) with intercept on log QMLE-Trade RV, p = 1..63, common hold-back of 63 observations.',fontsize=7.5,color=MUTED)
fig.savefig('pstar_monthly_vs_annual.png',dpi=180,bbox_inches='tight')
