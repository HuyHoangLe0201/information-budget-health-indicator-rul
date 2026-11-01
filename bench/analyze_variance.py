import numpy as np, pickle, json, sys
from scipy import stats
rng = np.random.default_rng(0); out = {}
def score(d): return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))
for FD in ['FD001', 'FD002', 'FD003', 'FD004']:
    r6 = pickle.load(open(f'out2/cmapss_{FD}_v6.pkl', 'rb')); r5 = pickle.load(open(f'out2/cmapss_{FD}_v5.pkl', 'rb')); y = r5['y']; n = len(y); d = {}
    rm = lambda med: float(np.sqrt(np.mean((med-y)**2)))
    for name in ['PGIB-HI', 'PGIB-KF-HI', 'PGIB-KFR-HI', 'beta-VAE-HI']:
        runs = [x for x in r6['methods'][name]]; per = np.array([rm(x['med']) for x in runs]); picp = np.array([100*np.mean((y >= x['lo']) & (y <= x['hi'])) for x in runs]); crps = np.array([np.mean(x['crps']) for x in runs])
        e = {'seed_rmse': per.tolist(), 'mean': float(per.mean()), 'sd': float(per.std(ddof=1)), 'min': float(per.min()), 'max': float(per.max()), 'picp': float(picp.mean()), 'crps': float(crps.mean())}
        if name+' (ens10)' in r6['methods']:
            x = r6['methods'][name+' (ens10)'][0]; e['ens'] = rm(x['med']); bs = [np.sqrt(np.mean((x['med'][i]-y[i])**2)) for i in [rng.integers(0, n, n) for _ in range(1500)]]; e['ens_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; e['ens_picp'] = float(100*np.mean((y >= x['lo']) & (y <= x['hi']))); e['ens_crps'] = float(np.mean(x['crps']))
        d[name] = e
    base = {k: r5['methods'][k][0] for k in ['Age-only', 'PCA-HI', 'Mon-opt HI', 'Ridge-HI (supervised)', 'Ridge-reg-HI']}; d['baselines'] = {k: rm(v['med']) for k, v in base.items()}
    # KF vs KFR across seeds (Welch) and ensemble paired bootstrap over engines
    a, b = np.array(d['PGIB-KF-HI']['seed_rmse']), np.array(d['PGIB-KFR-HI']['seed_rmse']); t = stats.ttest_ind(b, a, equal_var=False); df = len(a)+len(b)-2; se = np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b)); d['KFR_minus_KF'] = {'diff': float(b.mean()-a.mean()), 'ci95': [float(b.mean()-a.mean()-2.1*se), float(b.mean()-a.mean()+2.1*se)], 'p': float(t.pvalue)}
    def paired(m1, m2):
        e1 = (m1-y)**2; e2 = (m2-y)**2; ds = []
        for _ in range(3000): i = rng.integers(0, n, n); ds.append(np.sqrt(e1[i].mean())-np.sqrt(e2[i].mean()))
        return [float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    ek = r6['methods']['PGIB-KF-HI (ens10)'][0]['med']; er = r6['methods']['PGIB-KFR-HI (ens10)'][0]['med']; d['paired_ens'] = {'KFR_vs_KF': paired(er, ek)}
    for k, v in base.items():
        if k != 'Age-only': d['paired_ens'][f'KF_vs_{k}'] = paired(ek, v['med']); d['paired_ens'][f'KFR_vs_{k}'] = paired(er, v['med'])
    d['frac_seeds_below_best_baseline'] = {nm: float(np.mean(np.array(d[nm]['seed_rmse']) < min(d['baselines'][k] for k in ['PCA-HI', 'Mon-opt HI', 'Ridge-HI (supervised)', 'Ridge-reg-HI']))) for nm in ['PGIB-KF-HI', 'PGIB-KFR-HI']}
    out[FD] = d
    print(FD, 'baselines', {k: round(v, 1) for k, v in d['baselines'].items()})
    for nm in ['PGIB-HI', 'PGIB-KF-HI', 'PGIB-KFR-HI', 'beta-VAE-HI']: e = d[nm]; print('  %-12s single %.1f ± %.1f [%.1f, %.1f]' % (nm, e['mean'], e['sd'], e['min'], e['max']), ('ens10 %.1f [%.1f, %.1f]' % (e['ens'], *e['ens_ci'])) if 'ens' in e else '')
    print('  KFR-KF (seed means): %.1f, 95%% CI [%.1f, %.1f], Welch p=%.3f' % (d['KFR_minus_KF']['diff'], *d['KFR_minus_KF']['ci95'], d['KFR_minus_KF']['p']), '| ens KFR vs KF paired:', [round(x, 1) for x in d['paired_ens']['KFR_vs_KF']])
    print('  ens KF vs best simple (paired):', {k.replace('KF_vs_', ''): [round(x, 1) for x in v] for k, v in d['paired_ens'].items() if k.startswith('KF_vs_')})
json.dump(out, open('../results/variance10.json', 'w'), indent=1)
