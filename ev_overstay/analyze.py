"""
Result analysis and plots (Sections 6.3 to 6.7 of the paper).

  python -m ev_overstay.analyze results.csv --out figs/

Outputs:
  summary.csv            mean and standard deviation of each indicator for each combination
  table_methods.csv      method comparison table (mean ± standard deviation)
  wilcoxon.csv           paired Wilcoxon signed-rank tests of RH against the other methods
  fig_norm_reward.png/svg    box plots of the normalized reward (one panel per capacity ratio)
  fig_rules.png/svg          service rate against mean idle fee per owner under each rule
  fig_sens_*.png/svg         sensitivity curves (fee rate / grace period / quantile, if several values are present)
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'SimHei', 'Microsoft YaHei',
                                   'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

GROUP = ['behavior', 'n', 'ratio', 'rule', 'theta', 'phi', 'grace_h', 'quantile_p', 'notice']
METHOD_ORDER = ['OFF', 'B0', 'B1', 'RH-N', 'RH']
RULE_NAME = {'none': 'No fee', 'always': 'Always-on', 'congestion': 'Congestion-triggered'}
METRICS = ['reward', 'norm_reward', 'accept_rate', 'service_rate', 'served_of_accepted', 'avg_queue_h',
           'avg_overstay_h', 'block_h', 'reminders', 'night_reminders', 'avg_fee', 'fee_p95', 'max_fee', 'fee_payers',
           'energy_cost', 'utilization', 'abandoned', 'min_energy_violations', 'avg_solve_s', 'max_solve_s']


def rule_label(r):
    if r['rule'] == 'congestion':
        return f"Congestion-triggered (θ={int(r['theta'])})"
    return RULE_NAME.get(r['rule'], r['rule'])


def save(fig, out, name):
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), bbox_inches='tight', dpi=300 if ext == 'png' else None)
    plt.close(fig)


def load(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if 'behavior' not in df.columns:
        df['behavior'] = 'base'
    df['rule_label'] = df.apply(rule_label, axis=1)
    off = df[df.method == 'OFF'][GROUP + ['seed', 'reward']].rename(columns={'reward': 'off_reward'})
    if len(off):
        # the offline solution depends only on the instance and rule, not on notices or quantiles, so match by instance and rule
        keys = ['behavior', 'n', 'ratio', 'rule', 'theta', 'phi', 'grace_h', 'seed']
        off = off.drop_duplicates(keys)[keys + ['off_reward']]
        df = df.merge(off, on=keys, how='left')
        df['norm_reward'] = df['reward'] / df['off_reward']
    else:
        df['norm_reward'] = np.nan
    return df


def summary(df, out):
    cols = [c for c in METRICS if c in df.columns]
    g = df.groupby(GROUP + ['rule_label', 'method'])[cols]
    s = g.agg(['mean', 'std']).round(4)
    s.columns = [f'{a}_{b}' for a, b in s.columns]
    s.reset_index().to_csv(os.path.join(out, 'summary.csv'), index=False, encoding='utf-8-sig')

    show = ['norm_reward', 'accept_rate', 'service_rate', 'avg_queue_h', 'avg_overstay_h',
            'night_reminders', 'avg_fee', 'fee_p95', 'avg_solve_s']
    show = [c for c in show if c in df.columns]
    rows = []
    for key, sub in df.groupby(['ratio', 'rule_label', 'method']):
        r = dict(ratio=key[0], rule=key[1], method=key[2])
        for c in show:
            v = sub[c].dropna()
            r[c] = f'{v.mean():.3f}±{v.std():.3f}' if len(v) else ''
        rows.append(r)
    t = pd.DataFrame(rows)
    t['_o'] = t['method'].map({m: i for i, m in enumerate(METHOD_ORDER)})
    t.sort_values(['ratio', 'rule', '_o']).drop(columns='_o').to_csv(
        os.path.join(out, 'table_methods.csv'), index=False, encoding='utf-8-sig')


def tests(df, out, ref='RH', metric=None):
    metric = metric or ('norm_reward' if df['norm_reward'].notna().any() else 'reward')
    rows = []
    for key, sub in df.groupby(GROUP):
        piv = sub.pivot_table(index='seed', columns='method', values=metric)
        if ref not in piv:
            continue
        for m in piv.columns:
            if m in (ref, 'OFF'):
                continue
            d = (piv[ref] - piv[m]).dropna()
            if len(d) < 5 or np.allclose(d, 0):
                p = np.nan
            else:
                p = wilcoxon(d).pvalue
            rows.append(dict(zip(GROUP, key), compare=f'{ref} vs {m}', metric=metric,
                             mean_diff=d.mean(), win=(d > 0).sum(), lose=(d < 0).sum(), p_value=p))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'wilcoxon.csv'), index=False, encoding='utf-8-sig')


def fig_norm_reward(df, out):
    if not df['norm_reward'].notna().any():
        return
    d = df[df.method != 'OFF']
    ratios = sorted(d.ratio.unique())
    rules = d.rule_label.unique()
    for rl in rules:
        dd = d[d.rule_label == rl]
        fig, axes = plt.subplots(1, len(ratios), figsize=(3.2 * len(ratios), 3.0), sharey=True, squeeze=False)
        for ax, ra in zip(axes[0], ratios):
            sub = dd[dd.ratio == ra]
            ms = [m for m in METHOD_ORDER if m in sub.method.unique()]
            ax.boxplot([sub[sub.method == m]['norm_reward'].dropna() for m in ms],
                       widths=0.55, showmeans=True,
                       meanprops=dict(marker='^', markerfacecolor='k', markeredgecolor='k', markersize=4))
            ax.set_xticks(range(1, len(ms) + 1))
            ax.set_xticklabels(ms)
            ax.set_title(f'Capacity ratio {ra}', fontsize=10)
            ax.grid(axis='y', alpha=.3)
        axes[0][0].set_ylabel('Normalized reward (online / offline)')
        fig.suptitle(rl, fontsize=10)
        fig.tight_layout()
        save(fig, out, f'fig_norm_reward_{rl}')


def fig_rules(df, out, method='RH'):
    d = df[df.method == method]
    if d.rule_label.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    g = d.groupby(['ratio', 'rule_label'])[['service_rate', 'avg_fee', 'avg_overstay_h', 'night_reminders']].mean()
    g = g.reset_index()
    marks = ['o', 's', '^', 'D', 'v', 'P']
    for i, (rl, sub) in enumerate(g.groupby('rule_label')):
        axes[0].plot(sub['avg_fee'], sub['service_rate'], marker=marks[i % 6], ls='-', label=rl)
        for _, r in sub.iterrows():
            axes[0].annotate(f"{r['ratio']}", (r['avg_fee'], r['service_rate']), fontsize=7,
                             xytext=(3, 3), textcoords='offset points')
    axes[0].set_xlabel('Mean idle fee per accepted vehicle (CNY)')
    axes[0].set_ylabel('Service rate (vehicles reaching target / requests)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3)
    piv = g.pivot(index='ratio', columns='rule_label', values='avg_overstay_h')
    piv.plot.bar(ax=axes[1], rot=0, width=.75)
    axes[1].set_xlabel('Capacity ratio')
    axes[1].set_ylabel('Mean charger occupancy (h)')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=.3)
    fig.tight_layout()
    save(fig, out, f'fig_rules_{method}')


def fig_sens(df, out, col, xlabel, method='RH'):
    d = df[df.method == method]
    if d[col].nunique() < 2:
        return
    g = d.groupby(['ratio', col])[['service_rate', 'avg_fee', 'avg_overstay_h']].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ra, sub in g.groupby('ratio'):
        for ax, y in zip(axes, ['service_rate', 'avg_fee', 'avg_overstay_h']):
            ax.plot(sub[col], sub[y], marker='o', label=f'Capacity ratio {ra}')
    for ax, yl in zip(axes, ['Service rate', 'Mean idle fee (CNY)', 'Mean charger occupancy (h)']):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(yl)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, out, f'fig_sens_{col}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='+')
    ap.add_argument('--out', default='figs')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df = pd.concat([load(p) for p in a.csv], ignore_index=True)
    summary(df, a.out)
    tests(df, a.out)
    fig_norm_reward(df, a.out)
    fig_rules(df, a.out)
    fig_sens(df, a.out, 'phi', 'Fee rate (CNY/h)')
    fig_sens(df, a.out, 'grace_h', 'Grace period (h)')
    fig_sens(df, a.out, 'quantile_p', 'Delay quantile p')
    fig_sens(df, a.out, 'behavior', 'Move-out behavior scenario')
    print('Output folder:', a.out)


if __name__ == '__main__':
    main()
