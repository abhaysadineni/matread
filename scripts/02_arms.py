import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _common import args
_a = args(needs_mat=False)
OUT = _a.out

import numpy as np, h5py, json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score



m = np.load(OUT + '/meta2.npz', allow_pickle=True)
fs, treated, unit, strain, label, anti, std_nm = (m['fs'], m['treated'], m['unit'],
    m['strain'], m['label'], m['anti'], m['std_nm'])
f = h5py.File(OUT + '/nanomotion.h5', 'r')
rng = np.random.default_rng(7)

def bandpower(x, srate, nb=8):
    x = x - x.mean(); n = 1 << int(np.floor(np.log2(x.size)))
    P = (np.abs(np.fft.rfft(x[:n]*np.hanning(n)))**2)/n
    fr = np.fft.rfftfreq(n, 1/srate)
    e = np.geomspace(max(0.5, srate/n*4), srate/2*0.95, nb+1)
    return np.array([np.log10(P[(fr>=lo)&(fr<hi)].mean()+1e-20)
                     if ((fr>=lo)&(fr<hi)).any() else -20.0 for lo,hi in zip(e[:-1],e[1:])])

all_i = np.arange(len(fs))
rows, shared_total = [], 0
for e in sorted(set(label.tolist())):
    i = all_i[label == e]
    if not (treated[i].any() and (~treated[i]).any()):
        continue
    dT = set(unit[i][treated[i]].tolist()); dU = set(unit[i][~treated[i]].tolist())
    sh = dT & dU
    shared_total += len(sh)
    # keep only recordings on drums that carry BOTH classes: a true paired design
    keep = i[np.isin(unit[i], list(sh))]
    yv = treated[keep].astype(int); grp = unit[keep]
    folds = min(5, len(sh), int(min(np.bincount(yv))))
    if folds < 2:
        rows.append(dict(exp=e, n=len(keep), drums=len(sh), note='too few folds')); continue
    X = np.stack([bandpower(f['traces'][f'{k:04d}'][:], fs[k]) for k in keep])
    S = np.log10(std_nm[keep] + 1e-9).reshape(-1, 1)
    out = {}
    for nm, M in [('amp', S), ('spec', X)]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pr = cross_val_predict(pipe, M, yv, cv=GroupKFold(folds), groups=grp,
                               method='predict_proba')[:, 1]
        out[nm] = roc_auc_score(yv, pr)
    nl = []
    for _ in range(20):
        yp = rng.permutation(yv)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pr = cross_val_predict(pipe, X, yp, cv=GroupKFold(folds), groups=grp, method='predict_proba')[:,1]
        nl.append(roc_auc_score(yp, pr))
    rows.append(dict(exp=e, compound=sorted({a for a in anti[keep] if a not in ('','None')}),
                     strain=sorted(set(strain[keep].tolist())), fs=round(float(np.median(fs[keep])),1),
                     n=int(len(keep)), T=int(yv.sum()), U=int((1-yv).sum()), drums=len(sh),
                     amp=round(out['amp'],3), spec=round(out['spec'],3),
                     null_mean=round(float(np.mean(nl)),3), null_p95=round(float(np.percentile(nl,95)),3)))

print(f'drums carrying a within-experiment before/after pair: {shared_total}\n')
hdr = f"{'experiment':36s} {'fs':>7s} {'n':>4s} {'T':>4s} {'U':>4s} {'drums':>5s} {'amp':>6s} {'spec':>6s} {'null':>6s} {'p95':>6s}"
print(hdr); print('-'*len(hdr))
for r in sorted(rows, key=lambda x: -x.get('n',0)):
    if 'spec' not in r:
        print(f"{r['exp']:36s} {'':>7s} {r['n']:>4d} {'':>4s} {'':>4s} {r['drums']:>5d}  {r['note']}"); continue
    print(f"{r['exp']:36s} {r['fs']:>7.1f} {r['n']:>4d} {r['T']:>4d} {r['U']:>4d} {r['drums']:>5d} "
          f"{r['amp']:>6.3f} {r['spec']:>6.3f} {r['null_mean']:>6.3f} {r['null_p95']:>6.3f}")
json.dump(rows, open(OUT + '/arms.json','w'), indent=1)
