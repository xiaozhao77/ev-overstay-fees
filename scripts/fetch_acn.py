"""
Download ACN-Data sessions and summarize the connected time after charging is complete
(used in Section 6.2.2 of the paper: overnight sessions starting after 4 p.m.).

Usage (from the repository root):
    export ACN_TOKEN=<your token from https://ev.caltech.edu/dataset>
    python scripts/fetch_acn.py
Outputs acn_sessions.csv (all sessions) and acn_idle_summary.txt (summary). Downloaded pages are
cached in acn_cache/, so an interrupted run resumes. The raw ACN-Data are not redistributed here;
see the ACN-Data terms of use.
"""
import os, json
import time, requests, pandas as pd, numpy as np

TOKEN = os.environ.get('ACN_TOKEN', '')   # request a free token at https://ev.caltech.edu/dataset
if not TOKEN:
    raise SystemExit('Set the environment variable ACN_TOKEN first (free token at https://ev.caltech.edu/dataset)')
BASE = 'https://ev.caltech.edu/api/v1/'
SITES = ['caltech', 'jpl', 'office001']

CACHE = 'acn_cache'; os.makedirs(CACHE, exist_ok=True)
PER = 100                       # the server returns at most 100 records per page
skipped = []

def get_page(site, page):
    fn = os.path.join(CACHE, f'{site}_{page:05d}.json')
    if os.path.exists(fn):                       # pages already downloaded are read from the cache (resume)
        return json.load(open(fn))
    url = f'{BASE}sessions/{site}?max_results={PER}&page={page}'
    for k in range(8):
        try:
            r = requests.get(url, auth=(TOKEN, ''), timeout=60); r.raise_for_status()
            js = r.json(); json.dump(js, open(fn, 'w')); return js
        except Exception as e:
            w = min(10 * (k + 1), 60); print(f'  page {page} retry {k + 1} (wait {w} s): {str(e)[:80]}'); time.sleep(w)
    return None                                  # skip the page after repeated failures

rows = []
for site in SITES:
    first = get_page(site, 1)
    total = first['_meta']['total'] if first and '_meta' in first else None
    npage = (total + PER - 1) // PER if total else 10 ** 6
    print(site, 'total', total, 'records,', npage, 'pages', flush=True)
    for page in range(1, npage + 1):
        js = first if page == 1 else get_page(site, page)
        if js is None:
            skipped.append((site, page)); print('  skipped', site, 'page', page, 'after repeated failures'); continue
        items = js.get('_items', [])
        if not items: break
        for it in items:
            rows.append({'site': site, 'sessionID': it.get('sessionID'), 'userID': it.get('userID'),
                         'connectionTime': it.get('connectionTime'), 'disconnectTime': it.get('disconnectTime'),
                         'doneChargingTime': it.get('doneChargingTime'), 'kWhDelivered': it.get('kWhDelivered'),
                         'timezone': it.get('timezone')})
        if page % 20 == 0: print(f'  {site} page {page}/{npage}, {len(rows)} records so far', flush=True)
        time.sleep(0.2)
print('Skipped pages:', skipped if skipped else 'none')
d = pd.DataFrame(rows).drop_duplicates('sessionID')
d.to_csv('acn_sessions.csv', index=False)
for c in ['connectionTime', 'disconnectTime', 'doneChargingTime']:
    d[c] = pd.to_datetime(d[c], utc=True, errors='coerce')
tz = 'America/Los_Angeles'
d['conn_local'] = d.connectionTime.dt.tz_convert(tz)
d['stay_h'] = (d.disconnectTime - d.connectionTime).dt.total_seconds() / 3600
d['charge_h'] = (d.doneChargingTime - d.connectionTime).dt.total_seconds() / 3600
d['idle_h'] = (d.disconnectTime - d.doneChargingTime).dt.total_seconds() / 3600
ok = d.dropna(subset=['idle_h'])
ok = ok[(ok.idle_h >= 0) & (ok.stay_h > 0) & (ok.stay_h <= 24) & (ok.kWhDelivered > 0)]
hr = ok.conn_local.dt.hour + ok.conn_local.dt.minute / 60
groups = {'All sessions': ok,
          'Connected after 4 p.m.': ok[hr >= 16],
          'Connected after 4 p.m. and staying past midnight': ok[(hr >= 16) & (ok.disconnectTime.dt.tz_convert(tz).dt.date > ok.conn_local.dt.date)]}
lines = []
for name, g in groups.items():
    q = g.idle_h.quantile([0.25, 0.5, 0.75, 0.9]).round(2).tolist()
    lines.append(f'{name}: n={len(g)}, mean stay {g.stay_h.mean():.2f} h, mean charging {g.charge_h.mean():.2f} h, '
                 f'mean connected time after full charge {g.idle_h.mean():.2f} h (median {q[1]}, quartiles {q[0]}-{q[2]}, 90th percentile {q[3]}), '
                 f'idle share of stay {(g.idle_h.sum() / g.stay_h.sum()):.1%}')
for s in SITES:
    g = ok[ok.site == s]
    lines.append(f'  {s}: n={len(g)}, mean connected time after full charge {g.idle_h.mean():.2f} h')
open('acn_idle_summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
