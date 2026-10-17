"""
Figures 3-11 of the paper.

Usage (from the repository root):
    python scripts/make_all_figs.py
Reads the result files in results/ and writes 300-dpi PNG files to figures/.
Figures 7 and 9 use values from Table 11 and Table B4 that are written in this script.
The paper figures use Times New Roman; matplotlib falls back to another serif font if it is not installed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'results')
if not os.path.isdir(DATA):
    raise SystemExit('results/ folder not found; run this script from a copy of the repository')
F_METHODS = os.path.join(DATA, 'res_methods.csv')
F_N = os.path.join(DATA, 'res_n.csv')
F_PHI = os.path.join(DATA, 'res_phi.csv')
F_TOU = os.path.join(DATA, 'r10_hist_tou.csv')
F_FLAT = os.path.join(DATA, 'r10_hist_flat.csv')
F_Q = os.path.join(DATA, 'r13_qhist.csv')
F_BD = os.path.join(DATA, 'r13_boundary.csv')
F_FULL = os.path.join(DATA, 'r14_full.csv')
F_FULLRH = os.path.join(DATA, 'r8_full_t10.csv')
OUT = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT, exist_ok=True)

# ---------------- fonts and global style ----------------
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # suppress font-lookup messages
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Times New Roman',
                     'mathtext.it': 'Times New Roman:italic',
                     'mathtext.bf': 'Times New Roman:bold',
                     'mathtext.fallback': 'stix'})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['hatch.linewidth'] = 0.6
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5,
                     'axes.edgecolor': '#555', 'axes.linewidth': 0.8,
                     'xtick.color': '#333', 'ytick.color': '#333'})

C = {'B0': '#eb6834', 'B1': '#eda100', 'RH-N': '#1baf7a', 'RH': '#2a78d6'}
ORDER = ['B0', 'B1', 'RH-N', 'RH']
RC = {'none': '#8a8a8a', 'always': '#eb6834', 'congestion': '#2a78d6'}
RN = {'none': 'No fee', 'always': 'Always-on', 'congestion': 'Congestion-triggered'}


def grid(ax):
    ax.grid(axis='y', color='#e5e5e5', lw=0.6)
    ax.set_axisbelow(True)


# =====================================================================
# method comparison: normalized reward = online reward / offline optimum of the same instance
# =====================================================================
d = pd.read_csv(F_METHODS)
off = d[d.method == 'OFF'][['ratio', 'rule', 'seed', 'reward']].rename(columns={'reward': 'off'})
x = d[d.method != 'OFF'].merge(off, on=['ratio', 'rule', 'seed'])
x['norm'] = x.reward / x.off
cg = x[x.rule == 'congestion']          # Figures 4 and 5 use the congestion-triggered rule only

# ---------------- Figure 4: box plots of the normalized reward (one panel per capacity ratio) ----------------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, ra in zip(axs, [0.4, 0.6, 0.8]):
    s = cg[cg.ratio == ra]
    data = [s[s.method == m].norm.values for m in ORDER]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showmeans=True,
                    medianprops=dict(color='#222', lw=1.2),
                    whiskerprops=dict(color='#555', lw=0.8), capprops=dict(color='#555', lw=0.8),
                    flierprops=dict(marker='o', ms=3, mfc='none', mec='#777', mew=0.6),
                    meanprops=dict(marker='^', ms=4, mfc='white', mec='#222', mew=0.8))
    for p, m in zip(bp['boxes'], ORDER):
        p.set_facecolor(C[m]); p.set_alpha(0.85); p.set_edgecolor('#333'); p.set_linewidth(0.8)
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(ORDER)
    ax.set_title(f'Peak charger ratio {ra}')
    grid(ax)
axs[0].set_ylabel('Online reward / offline optimum')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4.png'), dpi=300); plt.close(fig)


def ci(v):
    """Half-width of the 95% confidence interval; if v has a seed index or column (pooled capacity levels), average by night (seed) first"""
    if isinstance(v, pd.Series) and 'seed' in (v.index.names or []):
        v = v.groupby(level='seed').mean()
    v = np.asarray(v, float); n = len(v); return _st.t.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
def cic(df, col):
    """Half-width of the confidence interval clustered by night when capacity levels are pooled"""
    return ci(df.groupby('seed')[col].mean())
EB=dict(ecolor='#333', elinewidth=0.8, capsize=2.5, capthick=0.8)
# ---------------- Figure 5 ----------------
panels = [('avg_overstay_h', 'Mean overstay of charged EVs (h)', '%.1f'),
          ('night_reminders', 'Late-night reminders per night', '%.1f'),
          ('fee_p95', '95th percentile idle fee (CNY)', '%.0f')]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
for ax, (col, title, fmt) in zip(axs, panels):
    v = np.array([cg[cg.method == m][col].mean() for m in ORDER]); e = np.array([cic(cg[cg.method == m], col) for m in ORDER])
    ax.bar(range(4), v, width=0.62, color=[C[m] for m in ORDER], edgecolor='white', linewidth=1, yerr=e, error_kw=EB)
    for i_, val in enumerate(v):
        ax.text(i_ + 0.08, val + e[i_], fmt % val, ha='left', va='bottom', fontsize=7.5, color='#222')
    ax.set_xticks(range(4)); ax.set_xticklabels(ORDER); ax.set_title(title)
    ax.set_ylim(0, max(v + e) * 1.18 if max(v) > 0 else 1)
    grid(ax)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5.png'), dpi=300); plt.close(fig)
# ---------------- Figure 6 ----------------
_f1 = pd.read_csv(F_FULL); _f2 = pd.read_csv(F_FULLRH)
dd = pd.concat([_f1[_f1.method == 'B0-NR'], _f2[_f2.method == 'RH']])
dd['rule'] = dd.rule.astype(str).replace({'cong0': 'congestion'})
M6 = ['B0-NR', 'RH']
pv = dd.pivot_table(index=['method', 'ratio', 'seed'], columns='rule', values='reward')
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axs[0]
for k, ru in enumerate(['always', 'congestion']):
    vals, errs = [], []
    for m in M6:
        p_ = pv.loc[m]; diff = p_[ru] - p_['none']; b = p_['none'].mean()
        vals.append(100 * diff.mean() / b); errs.append(100 * ci(diff.groupby(level='seed').mean()) / b)
    xs = np.arange(2) + (k - 0.5) * 0.3
    ax.bar(xs, vals, width=0.28, color=RC[ru], edgecolor='white', linewidth=1, yerr=errs, error_kw=EB)
    for xx, vv, ee in zip(xs, vals, errs):
        ax.text(xx, vv + ee, '%.1f%%' % vv, ha='center', va='bottom', fontsize=7, color='#222')
ax.set_xticks(range(2)); ax.set_xticklabels(M6); ax.set_title('Reward gain over no fee (%)')
ax.set_ylim(0, 11); grid(ax)
for ax, (col, title) in zip(axs[1:], [('avg_fee', 'Mean idle fee (CNY)'), ('fee_p95', '95th percentile idle fee (CNY)')]):
    top = 0
    for k, ru in enumerate(['always', 'congestion']):
        vals = [dd[(dd.method == m) & (dd.rule == ru)][col].mean() for m in M6]
        errs = [cic(dd[(dd.method == m) & (dd.rule == ru)], col) for m in M6]
        xs = np.arange(2) + (k - 0.5) * 0.3
        ax.bar(xs, vals, width=0.28, color=RC[ru], edgecolor='white', linewidth=1, label=RN[ru], yerr=errs, error_kw=EB)
        for xx, vv, ee in zip(xs, vals, errs):
            ax.text(xx, vv + ee, '%.0f' % vv, ha='center', va='bottom', fontsize=7, color='#222')
        top = max(top, max(np.add(vals, errs)))
    ax.set_xticks(range(2)); ax.set_xticklabels(M6); ax.set_title(title)
    ax.set_ylim(0, top * 1.15); grid(ax)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=len(l), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=2.0)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, 'fig6.png'), dpi=300); plt.close(fig)
# ---------------- Figure 11 (requests per night and fee rate; 2x2: (a) reward gain and (b) mean fee against requests, (c) reward and (d) mean fee against fee rate) ----------------
N_ = pd.read_csv(F_N); P_ = pd.read_csv(F_PHI)
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
axs = axs.ravel(); ns = [12, 24, 36, 48]
ax = axs[0]
for m in ['B0', 'RH']:
    v, e = [], []
    for k in ns:
        p_ = N_[(N_.n == k) & (N_.method == m)].pivot(index='seed', columns='rule', values='reward')
        diff = p_['congestion'] - p_['none']; b = p_['none'].mean(); v.append(100 * diff.mean() / b); e.append(100 * ci(diff) / b)
    v, e = np.array(v), np.array(e)
    ax.plot(ns, v, 'o-', color=C[m], lw=1.6, ms=4); ax.fill_between(ns, v - e, v + e, color=C[m], alpha=0.15, lw=0)
ax.set_xticks(ns); ax.set_xlabel('Requests per night'); ax.set_title('(a) Reward gain of congestion-triggered rule (%)')
ax.set_ylim(-1, 13)
ax = axs[1]
for m in ['B0', 'RH']:
    for ru, ls in [('congestion', '-'), ('always', '--')]:
        s_ = [N_[(N_.n == k) & (N_.method == m) & (N_.rule == ru)].avg_fee for k in ns]
        v = np.array([x_.mean() for x_ in s_]); e = np.array([ci(x_) for x_ in s_])
        ax.errorbar(ns, v, yerr=e, fmt=ls, marker='o', ms=3.5, color=C[m], lw=1.4, capsize=2, elinewidth=0.7)
ax.set_xticks(ns); ax.set_xlabel('Requests per night'); ax.set_title('(b) Mean idle fee (CNY)')
phis = [30, 60, 120, 384]
for ax, col, title in [(axs[2], 'reward', '(c) Online reward'), (axs[3], 'avg_fee', '(d) Mean idle fee (CNY)')]:
    for m in ['B0', 'RH']:
        for ru, ls, lab in [('congestion', '-', 'congestion-triggered'), ('always', '--', 'always-on')]:
            s_ = [P_[(P_.phi == f) & (P_.method == m) & (P_.rule == ru)][col] for f in phis]
            v = np.array([x_.mean() for x_ in s_]); e = np.array([ci(x_) for x_ in s_])
            ax.errorbar(phis, v, yerr=e, fmt=ls, marker='o', ms=3.5, color=C[m], lw=1.4, label=f'{m} {lab}', capsize=2, elinewidth=0.7)
    ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(phis); ax.set_xticklabels([str(f) for f in phis]); ax.set_xlim(25, 460)
    ax.set_xlabel('Fee rate (CNY/h, log scale)'); ax.set_title(title)
for ax in axs: grid(ax)
h, l = axs[2].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=len(l), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.8)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, 'fig11.png'), dpi=300); plt.close(fig)


# ---------------- Figure 8: reward gap between the congestion-triggered and always-on rules under weaker responses (ratio 0.6, fee notices on; 90% confidence intervals) ----------------
BD = pd.read_csv(F_BD)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, m, title in zip(axs, ['B0', 'RH'], ['(a) B0', '(b) RH']):
    g = BD[(BD.method == m) & (BD.kind == 'g1')].sort_values('x')
    ax.fill_between(g.x, g.lo90, g.hi90, color=C[m], alpha=0.18, lw=0)
    ax.plot(g.x, g.gap, '-', color=C[m], lw=1.4)
    g100, g30 = g[g.n >= 100], g[g.n < 100]
    ax.plot(g100.x, g100.gap, 'o', color=C[m], ms=4.2)                      # 100 instances
    ax.plot(g30.x, g30.gap, 'o', mfc='white', mec=C[m], mew=1.2, ms=4.2)     # first 30 instances
    k = BD[(BD.method == m) & (BD.kind == 'ign')].sort_values('x')
    ax.errorbar(k.x, k.gap, yerr=[k.gap - k.lo90, k.hi90 - k.gap], fmt='s', ms=4.2, mfc='white', mec='#222', ecolor='#222', elinewidth=0.8, capsize=2.5)
    ax.axhline(-2, color='#555', lw=0.9, ls='--'); ax.axhline(-1, color='#555', lw=0.8, ls=':'); ax.axhline(0, color='#999', lw=0.6)
    ax.set_title(title); ax.set_xlim(-3, 88); ax.set_ylim(-8.5, 1)
    ax.set_xlabel('Reduction in daytime grace-period probability (%)'); grid(ax)
axs[0].set_ylabel('Reward difference to always-on rule (%)')
h10 = [Line2D([], [], color='#555', marker='o', ms=3.8, lw=1.4, label=r'Lower $\gamma_1$ (100 instances)'),
       Line2D([], [], color='#555', marker='o', mfc='white', ms=3.8, lw=1.4, label=r'Lower $\gamma_1$ (30 instances)'),
       Line2D([], [], color='#222', marker='s', mfc='white', ls='none', ms=4.2, label='Owners ignoring fee state'),
       Line2D([], [], color='#555', ls='--', lw=0.9, label='−2% margin'),
       Line2D([], [], color='#555', ls=':', lw=0.8, label='−1% margin')]
fig.legend(handles=h10, loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(os.path.join(OUT, 'fig8.png'), dpi=300); plt.close(fig)

# ---------------- Figure 7 (reward against mean fee, values from Table 11) ----------------
C7 = {'B0': '#eb6834', 'RH': '#2a78d6'}
# (theta, mean fee CNY, reward) from Table 11; queue = queue-triggered rule (20)
data = {
    'B0': {'q': (13, 1457.5),
           'th': [(0, 31, 1461.2), (1, 55, 1454.7), (2, 70, 1461.2), (3, 104, 1458.3), (4, 123, 1458.3),
                  (6, 148, 1461.2), (9, 161, 1464.1), (12, 166, 1464.1)]},
    'RH': {'q': (8, 1617.7),
           'th': [(0, 16, 1623.6), (1, 17, 1608.1), (2, 17, 1617.5), (3, 19, 1614.8), (4, 15, 1626.6),
                  (6, 19, 1611.8), (9, 22, 1604.8), (12, 19, 1612.1)]}}
# label offsets (avoid overlapping numbers)
OFF = {'B0': {9: (-6, 6), 12: (4, -10)}, 'RH': {6: (5, -9), 12: (5, 2), 3: (5, 5), 2: (-9, 4)}}

fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
for ax, m, title in zip(axs, ['B0', 'RH'], ['(a) B0', '(b) RH']):
    th = data[m]['th']
    x = [a[1] for a in th]; y = [a[2] for a in th]
    base = th[0][2]
    ax.axhspan(base * 0.98, base * 1.02, color='#d9d9d9', alpha=0.45, lw=0, zorder=0)
    if m == 'B0':
        ax.plot(x, y, '-', color=C7[m], lw=0.9, alpha=0.6, zorder=1)
    ax.scatter(x, y, s=22, color=C7[m], edgecolor='#333', lw=0.5, zorder=3)
    for k, xi, yi in th:
        lab = '12 (always-on)' if k == 12 else str(k)
        ax.annotate(lab, (xi, yi), textcoords='offset points', xytext=OFF[m].get(k, (3, 4)),
                    fontsize=7.5, color='#333')
    qx, qy = data[m]['q']
    ax.scatter([qx], [qy], s=34, marker='D', color='white', edgecolor='#222', lw=0.9, zorder=4)
    ax.set_title(title)
    ax.set_xlabel('Mean idle fee (CNY per accepted vehicle)')
    ax.set_ylabel('Online reward')
    ax.grid(color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(*((1420, 1500) if m == 'B0' else (1580, 1660)))
    ax.set_xlim(*((0, 185) if m == 'B0' else (0, 30)))
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

# ---------- legend below the plot, one row ----------
handles = [Line2D([], [], marker='o', ls='', markerfacecolor=C7['B0'], markeredgecolor='#333', markersize=5, label='Threshold θ, B0'),
           Line2D([], [], marker='o', ls='', markerfacecolor=C7['RH'], markeredgecolor='#333', markersize=5, label='Threshold θ, RH'),
           Line2D([], [], marker='D', ls='', markerfacecolor='white', markeredgecolor='#222', markersize=5, label='Queue-triggered (20)'),
           Patch(facecolor='#d9d9d9', alpha=0.45, label='±2% of reward at θ = 0')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.2)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUT, 'fig7.png'), dpi=300); plt.close(fig)




# ---------------- Figure 9: reward decomposition (change relative to B0 without fees; values from Table B4, mean of 90 instances, fee notices off) ----------------
rows8 = [  # label, admission reward, target reward, electricity, waiting cost, reminder cost, total reward
    ('B0\nno fee',         167.67, 1493.33, 238.18, 20.61, 52.94, 1349.27),
    ('B0\ncongestion',     173.89, 1607.78, 252.34, 19.96, 59.17, 1450.21),
    ('B0-NR\ncongestion',  173.78, 1533.33, 243.56, 21.49,  0.00, 1442.05),
    ('B0-TOU\ncongestion', 161.33, 1360.00, 196.90, 22.13,  0.00, 1302.30),
    ('B0-TOU+\ncongestion', 162.22, 1430.00, 207.72, 20.88, 10.83, 1352.79),
    ('RH\nno fee',         165.33, 1544.44, 192.13, 18.12,  0.00, 1499.53),
    ('RH\ncongestion',     172.00, 1666.67, 211.36, 15.83,  0.00, 1611.47)]
comp8 = [('Admission reward', 1, +1, '#9ecae1'), ('Target reward', 2, +1, '#2a78d6'),
         ('Electricity cost', 3, -1, '#eda100'), ('Waiting cost', 4, -1, '#8a8a8a'), ('Reminder cost', 5, -1, '#eb6834')]
ref8 = rows8[0]
fig, ax = plt.subplots(figsize=(7.0, 3.6))
for i, r in enumerate(rows8[1:]):
    pos, neg = 0.0, 0.0
    for name, k, sign, col in comp8:
        v = sign * (r[k] - ref8[k])          # change in the contribution to the total reward (cost terms negated, savings positive)
        if v >= 0:
            ax.bar(i, v, 0.6, bottom=pos, color=col, edgecolor='white', lw=0.4); pos += v
        else:
            ax.bar(i, v, 0.6, bottom=neg, color=col, edgecolor='white', lw=0.4); neg += v
    net = r[6] - ref8[6]
    ax.plot(i, net, marker='D', color='black', ms=5, zorder=5)
    ax.annotate(f'{net:+.0f}\n({r[6]:.0f})', (i, max(pos, net)), textcoords='offset points',
                xytext=(0, 4), ha='center', va='bottom', fontsize=8)
ax.axhline(0, color='#333', lw=0.8)
ax.set_xticks(range(len(rows8) - 1)); ax.set_xticklabels([r[0] for r in rows8[1:]])
ax.set_ylabel('Change in reward relative to B0 without fees'); ax.set_ylim(-230, 330); grid(ax)
h = [Patch(color=c, label=n) for n, _, _, c in comp8] + [Line2D([], [], marker='D', color='black', ls='none', ms=5, label='Net change')]
fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, 'fig9.png'), dpi=300); plt.close(fig)

# ---------------- Figure 10(a): vehicles reaching their targets per hour (congestion-triggered rule, 90 instances, fee notices off) ----------------
def hist9(f, m):
    h_ = pd.read_csv(f); h_ = h_[(h_.method == m) & (h_.rule == 'congestion')]
    return h_.full_hist.str.split(';', expand=True).astype(int).mean().values
series9 = [('B0 (time-of-use)', F_TOU, 'B0', C['B0'], 'o', '-'),
           ('B0-TOU (time-of-use)', F_TOU, 'B0-TOU', C['B1'], 's', '-'),
           ('RH (time-of-use)', F_TOU, 'RH', C['RH'], 'D', '-'),
           ('RH (flat tariff)', F_FLAT, 'RH', C['RH'], 'D', '--'),
           ('RH-N (time-of-use)', F_TOU, 'RH-N', C['RH-N'], '^', '-')]
hours9 = [(16 + i) % 24 for i in range(18)]; x9 = list(range(18))
HAS_Q = os.path.exists(F_Q)
if HAS_Q:
    fig = plt.figure(figsize=(6.6, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.16, 1, 0.62], hspace=0.08)
else:
    fig = plt.figure(figsize=(6.6, 3.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 1], hspace=0.03)
axh = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=axh)
# double arrows above mark the off-peak period (22:00-6:00) and the weak late-night response (22:00-7:00)
axh.set_ylim(0, 2); axh.axis('off')
for y_, x1, lab, col in [(1.25, 14, 'Off-peak tariff (22:00–6:00)', '#1f4e79'),
                         (0.35, 15, 'Late night, weak owner response (22:00–7:00)', '#333333')]:
    axh.annotate('', xy=(6, y_), xytext=(x1, y_), arrowprops=dict(arrowstyle='<->', color=col, lw=0.9, shrinkA=0, shrinkB=0))
    axh.text((6 + x1) / 2, y_ + 0.12, lab, ha='center', va='bottom', fontsize=7.5, color=col)
for xv in (6, 14, 15):
    ax.axvline(xv, color='#9a9a9a', lw=0.6, ls=':', zorder=1)
for lab, f, m, c, mk, ls in series9:
    ax.plot(x9, hist9(f, m), color=c, marker=mk, ms=3.5, lw=1.2, ls=ls, label=lab, zorder=3)
ax.set_ylim(-0.3, 12); ax.set_xlim(-0.5, 17.5); grid(ax)
ax.set_ylabel('Vehicles reaching\ntheir targets per night')
if HAS_Q:
    ax.text(0.005, 0.97, '(a)', transform=ax.transAxes, va='top', fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)
    axq = fig.add_subplot(gs[2], sharex=ax)
    q_ = pd.read_csv(F_Q); q_ = q_[q_.rule.astype(str).str.startswith('cong')]
    for m, mk in [('B0', 'o'), ('RH', 'D'), ('RH-N', '^')]:
        v = q_[q_.method == m].queue_hist.str.split(';', expand=True).astype(float).mean().values
        axq.plot(x9, v, color=C[m], marker=mk, ms=3.2, lw=1.2, zorder=3)
    for xv in (6, 14, 15):
        axq.axvline(xv, color='#9a9a9a', lw=0.6, ls=':', zorder=1)
    axq.set_ylabel('Vehicles waiting\n(mean per hour)'); grid(axq)
    axq.text(0.005, 0.95, '(b)', transform=axq.transAxes, va='top', fontsize=9)
    axq.set_ylim(bottom=0)
    ax_lab = axq
else:
    ax_lab = ax
ax_lab.set_xticks(x9); ax_lab.set_xticklabels([f'{h}:00' for h in hours9], rotation=45, ha='right', fontsize=7.5)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.0)
fig.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.14 if HAS_Q else 0.2)
fig.savefig(os.path.join(OUT, 'fig10.png'), dpi=300); plt.close(fig)

# ---------------- Figure 3 ----------------
# ---------------- Figure 3: states and system actions over four periods (schematic, no data) ----------------
from matplotlib.patches import FancyBboxPatch, Rectangle
def fig3():
    fig, ax = plt.subplots(figsize=(7.4, 4.75))
    ax.set_xlim(0, 100); ax.set_ylim(10.3, 72); ax.axis('off')
    DARK, EDGE = '#4a4a4a', '#222222'
    def box(x, y, w, h, text='', kind='charging', fs=8.5):
        """x, y = lower-left corner"""
        if kind == 'charging':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc=DARK, ec=EDGE, lw=0.8); tc = 'white'
        elif kind == 'not':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc='white', ec=EDGE, lw=0.8); tc = '#111'
        elif kind == 'full':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc='white', ec=EDGE, lw=0.8, hatch='///'); tc = '#111'
        elif kind == 'idle':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc='white', ec='#888', lw=0.8, ls='--'); tc = '#777'
        elif kind == 'remind':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.5', fc='#111', ec='#111', lw=0.8); tc = 'white'
        elif kind == 'defer':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.5', fc='white', ec=EDGE, lw=0.8, ls='--'); tc = '#111'
        elif kind == 'queue':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.5', fc='white', ec=EDGE, lw=0.9, ls='--'); tc = '#111'
        elif kind == 'req':
            p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.5', fc='white', ec=EDGE, lw=0.8); tc = '#111'
        ax.add_patch(p)
        if text:
            if kind == 'full':
                ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc,
                        bbox=dict(fc='white', ec='none', pad=0.4))
            else:
                ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc)
    # row positions
    Y = dict(title=68.5, event=63.0, req=57.0, queue=51.0, c1=43.5, c2=37.0, c3=30.5, fee=24.0, act=17.0)
    labels = [('Event', Y['event']), ('New\nrequests', Y['req']), ('Queue\n(K = 1)', Y['queue']),
              ('Fee\nstate', Y['fee']), ('Actions', Y['act'])]
    for t_, y_ in labels:
        ax.text(4.2, y_, t_, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(3.6, Y['c2'], 'Chargers', ha='center', va='center', fontsize=9, fontweight='bold')
    for c in ['c1', 'c2', 'c3']:
        ax.text(9.6, Y[c], c.upper(), ha='center', va='center', fontsize=9)
    X0 = [11.5, 34.0, 56.5, 79.0]; PW = 20.5
    titles = [1, 2, 3, 4]
    events = ['v1 to v4 arrive', 'v1 fully charged\nv5 arrives', 'v1 leaves, v4 plugs in\nv2 fully charged', 'v3 leaves at its\ndeclared departure']
    for k, x in enumerate(X0):
        ax.add_patch(FancyBboxPatch((x, 11.0), PW, 60.5, boxstyle='round,pad=0,rounding_size=1.2', fc='white', ec='#888', lw=0.9, zorder=0))
        ax.text(x + PW / 2, Y['title'], r'Period $t = %d$' % titles[k], ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x + PW / 2, Y['event'], events[k], ha='center', va='center', fontsize=8.5, linespacing=1.15)
        if k < 3:
            ax.annotate('', xy=(X0[k + 1] - 0.3, Y['c2']), xytext=(x + PW + 0.3, Y['c2']),
                        arrowprops=dict(arrowstyle='-|>', color='#222', lw=0.8, mutation_scale=7))
    bw, bh = 11.5, 4.0
    cx = lambda x: x + 1.6
    yb = lambda c: Y[c] - bh / 2
    # new requests
    for i, v in enumerate(['v1 √', 'v2 √', 'v3 √', 'v4 √']):
        box(X0[0] + 0.6 + i * 4.9, Y['req'] - 1.8, 4.5, 3.6, v, 'req', fs=8)
    box(X0[1] + PW / 2 - 2.4, Y['req'] - 1.8, 4.8, 3.6, 'v5 ×', 'req', fs=8)
    for k in (2, 3):
        ax.text(X0[k] + PW / 2, Y['req'], 'None', ha='center', va='center', fontsize=8.5, color='#777')
    # waiting area
    for k, txt in enumerate(['v4', 'v4', '', '']):
        box(X0[k] + PW / 2 - 3.2, Y['queue'] - 1.7, 6.4, 3.4, txt, 'queue', fs=8.5)
    # chargers
    S = [  # (c1, c2, c3, right-hand label)
        [('v1  charging', 'charging'), ('v2  charging', 'charging'), ('v3 not charging', 'not'), None],
        [('v1  full', 'full'), ('v2  charging', 'charging'), ('v3  charging', 'charging'), ('c1', 'Remind', 'remind')],
        [('v4  charging', 'charging'), ('v2  full', 'full'), ('v3  charging', 'charging'), ('c2', 'Defer', 'defer')],
        [('v4  charging', 'charging'), ('v2  full', 'full'), ('idle', 'idle'), ('c2', 'Defer', 'defer')]]
    for k, x in enumerate(X0):
        for c, (txt, kind) in zip(['c1', 'c2', 'c3'], S[k][:3]):
            box(cx(x), yb(c), bw, bh, txt, kind, fs=8 if txt == 'v3 not charging' else 8.5)
        if S[k][3]:
            c, txt, kind = S[k][3]
            box(cx(x) + bw + 0.8, Y[c] - 1.6, 5.6, 3.2, txt, kind, fs=7.5)
    # fee state
    fees = [(0, 1), (0, 1), (0, 1), (1, 0)]
    for k, (idle, kap) in enumerate(fees):
        ax.text(X0[k] + PW / 2, Y['fee'], r'Idle chargers %d $\rightarrow$ $\kappa_t = %d$' % (idle, kap),
                ha='center', va='center', fontsize=8.5)
    acts = ['Accept v1 to v4\nCharge v1, v2\nv3 waits for capacity',
            'Reject v5 (queue full)\nCharge v2, v3\nRemind v1',
            'Charge v3, v4\nDefer reminder for v2\n(no vehicle waiting)',
            'Charge v4\nKeep deferring v2\n(idle charger, no fee)']
    for k, a in enumerate(acts):
        ax.text(X0[k] + PW / 2, Y['act'], a, ha='center', va='center', fontsize=8.5, linespacing=1.2)
    # legend (below, one row)
    from matplotlib.patches import Patch
    h = [Patch(fc=DARK, ec=EDGE, label='Charging'), Patch(fc='white', ec=EDGE, label='Not charging'),
         Patch(fc='white', ec=EDGE, hatch='///', label='Fully charged (overstay)'),
         Patch(fc='#111', ec='#111', label='Reminder sent'), Patch(fc='white', ec=EDGE, ls='--', label='Reminder deferred')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.06),
               handlelength=1.8, handleheight=1.1, columnspacing=1.2)
    fig.text(0.5, 0.012, '3 chargers and 1 waiting space; during the evening peak the residual transformer capacity supports only 2 chargers.\n'
             r'Congestion-triggered rule with $\theta = 0$ and a grace period of one period; √ accepted, × rejected.',
             ha='center', va='bottom', fontsize=7.5, linespacing=1.3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.135)
    fig.savefig(os.path.join(OUT, 'fig3.png'), dpi=300); plt.close(fig)

fig3()

print('Figures 3-11 saved to', OUT)
