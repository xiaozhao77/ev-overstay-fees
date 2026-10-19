"""
Figure 12: served vehicles against late-night reminders (Section 6.7.5).

(a) Congestion-triggered rule, reminder clock, fee notices off, 90 instances over three capacity levels.
(b) Ratio 0.6, fee notices on, 30 instances, without early departures (hollow) and with 30% of owners
    leaving 0.5 to 2 h before their declared departure (filled). With a 2 h margin the plan never relies on the
    last 2 h of a stay, so RH with the margin gives identical results with and without early departures.
Writes figures/fig12.png (300 dpi). Run from the repository root.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.family'] = ['Times New Roman', 'Liberation Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.edgecolor': '#555', 'axes.linewidth': 0.8,
                     'xtick.color': '#333', 'ytick.color': '#333'})

R = 'results'
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
C = {'B0': '#eb6834', 'B0-NR': '#8c6bb1', 'B0-TOU': '#a6a6a6', 'B0-TOU+': '#5f5f5f',
     'B0-Q': '#1baf7a', 'RH': '#2a78d6', 'RH, 2 h margin': '#2a78d6'}
MK = {'B0': 'o', 'B0-NR': 's', 'B0-TOU': 'v', 'B0-TOU+': '^', 'B0-Q': 'D', 'RH': 'o', 'RH, 2 h margin': 'P'}


def summary(df, night_col):
    """Mean served vehicles with a 95% CI over nights (night means when capacity levels are pooled) and mean late-night reminders."""
    g = df.groupby('seed')[['served', night_col]].mean()
    n = len(g)
    h = tdist.ppf(0.975, n - 1) * g.served.std(ddof=1) / np.sqrt(n)
    return g[night_col].mean(), g.served.mean(), h


fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))

# ---------------- (a)
ax = axs[0]
rm = pd.read_csv(os.path.join(R, 'res_methods.csv'))
tp = pd.read_csv(os.path.join(R, 'r12_tou_plus.csv'))
bq = pd.read_csv(os.path.join(R, 'r15_b0q.csv'))
pts = {
    'B0': summary(rm[(rm.method == 'B0') & (rm.rule == 'congestion')], 'night_reminders'),
    'B0-NR': summary(tp[(tp.method == 'B0-NR') & (tp.rule == 'congestion')], 'night_reminders'),
    'B0-TOU': summary(tp[(tp.method == 'B0-TOU') & (tp.rule == 'congestion')], 'night_reminders'),
    'B0-TOU+': summary(tp[(tp.method == 'B0-TOU+') & (tp.rule == 'congestion')], 'night_reminders'),
    'B0-Q': summary(bq[(bq.method == 'B0-Q') & (bq.rule == 'congestion') & (bq.clock == 'rem')], 'night_rem'),
    'RH': summary(rm[(rm.method == 'RH') & (rm.rule == 'congestion')], 'night_reminders'),
}
fx = [pts['B0-NR'][0], pts['B0-Q'][0], pts['B0'][0]]
fy = [pts['B0-NR'][1], pts['B0-Q'][1], pts['B0-Q'][1]]
ax.plot(fx[:2], fy[:2], '--', color='#999', lw=0.9, zorder=1)
off = {'B0': (-0.4, -0.35), 'B0-NR': (0.4, -0.3), 'B0-TOU': (0.4, -0.1), 'B0-TOU+': (0.4, -0.1),
       'B0-Q': (0.35, -0.38), 'RH': (0.4, 0.12)}
for m, (x, y, h) in pts.items():
    ax.errorbar(x, y, yerr=h, fmt=MK[m], color=C[m], ms=6, capsize=2.5, elinewidth=0.8, zorder=3, label=m)
    ax.text(x + off[m][0], y + off[m][1], m, fontsize=8, color='#222', ha='left' if off[m][0] > 0 else 'right')
ax.set_xlabel('Late-night reminders per night')
ax.set_ylabel('Served vehicles per night')
ax.set_title('(a) Congestion-triggered rule, 90 instances')
ax.set_xlim(-1, 13.5)
ax.set_ylim(13.0, 17.5)
ax.grid(color='#e5e5e5', lw=0.6)
ax.set_axisbelow(True)

# ---------------- (b)
ax = axs[1]
rn = pd.read_csv(os.path.join(R, 'res_n.csv'))
rn = rn[(rn.n == 24) & (rn.method == 'RH') & (rn.rule == 'congestion')]
rb0 = pd.read_csv(os.path.join(R, 'r19_rb_e0.csv'))
rb3 = pd.read_csv(os.path.join(R, 'r19_rb_e0.3.csv'))
rh3 = pd.read_csv(os.path.join(R, 'r19_rh_e0.3.csv'))
rm0 = pd.read_csv(os.path.join(R, 'r20_rh_b2_e0.csv'))
rm3 = pd.read_csv(os.path.join(R, 'r20_rh_b2_e0.3.csv'))
pairs = {
    'B0': (rb0[(rb0.method == 'B0') & (rb0.rule == 'congestion')], rb3[(rb3.method == 'B0') & (rb3.rule == 'congestion')]),
    'B0-Q': (rb0[(rb0.method == 'B0-Q') & (rb0.rule == 'congestion')], rb3[(rb3.method == 'B0-Q') & (rb3.rule == 'congestion')]),
    'RH': (rn, rh3[rh3.rule == 'congestion']),
    'RH, 2 h margin': (rm0, rm3),
}
dx = {'B0': 0.0, 'B0-Q': 0.0, 'RH': -0.15, 'RH, 2 h margin': 0.15}
for m, (a0, a3) in pairs.items():
    x0, y0, h0 = summary(a0, 'night_reminders')
    x3, y3, h3 = summary(a3, 'night_reminders')
    x0 += dx[m]; x3 += dx[m]
    ax.errorbar(x0, y0, yerr=h0, fmt=MK[m], mfc='white', mec=C[m], color=C[m], ms=6, capsize=2.5, elinewidth=0.8, zorder=3)
    ax.errorbar(x3, y3, yerr=h3, fmt=MK[m], color=C[m], ms=6, capsize=2.5, elinewidth=0.8, zorder=3)
    if abs(y3 - y0) > 0.05:
        ax.annotate('', xy=(x3, y3), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=C[m], lw=0.9), zorder=2)
ax.text(0.5, 16.92, 'RH', fontsize=8, color='#222', va='center')
ax.text(0.5, 15.23, 'RH, 30% early', fontsize=8, color='#222', va='center')
ax.text(0.5, 16.40, 'RH, 2 h margin', fontsize=8, color='#222', va='center')
ax.text(3.0, 15.82, 'B0-Q', fontsize=8, color='#222')
ax.text(11.4, 15.75, 'B0', fontsize=8, color='#222', ha='right')
ax.set_xlabel('Late-night reminders per night')
ax.set_ylabel('Served vehicles per night')
ax.set_title('(b) Early departures, ratio 0.6, 30 instances')
ax.set_xlim(-1, 13.5)
ax.set_ylim(13.0, 17.5)
ax.grid(color='#e5e5e5', lw=0.6)
ax.set_axisbelow(True)

# legend below, one row
from matplotlib.lines import Line2D
hs = [Line2D([0], [0], marker=MK[m], color=C[m], lw=0, ms=6, label=m) for m in ['B0', 'B0-NR', 'B0-TOU', 'B0-TOU+', 'B0-Q', 'RH']]
hs += [Line2D([0], [0], marker='P', color=C['RH'], lw=0, ms=6, label='RH, 2 h margin'),
       Line2D([0], [0], marker='o', mfc='white', mec='#555', color='#555', lw=0, ms=6, label='No early departures (b)')]
fig.legend(handles=hs, loc='lower center', ncol=len(hs), frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0),
           handlelength=1.0, columnspacing=0.9, handletextpad=0.3)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, 'fig12.png'), dpi=300)
plt.close(fig)
for m, (x, y, h) in pts.items():
    print('(a)', m, round(x, 2), round(y, 2), round(h, 2))
