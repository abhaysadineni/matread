import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _common import args
_a = args(needs_mat=False)
OUT = _a.out

import numpy as np, h5py
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score



m = np.load(OUT + '/meta2.npz', allow_pickle=True)
fs, treated, unit, strain, label = m['fs'], m['treated'], m['unit'], m['strain'], m['label']
f = h5py.File(OUT + '/nanomotion.h5', 'r')
rng = np.random.default_rng(0)

def bandpower(x, srate, nb=8):
    x = x - x.mean(); n = 1 << int(np.floor(np.log2(x.size)))
    P = (np.abs(np.fft.rfft(x[:n]*np.hanning(n)))**2)/n
    fr = np.fft.rfftfreq(n, 1/srate)
    e = np.geomspace(max(0.5, srate/n*4), srate/2*0.95, nb+1)
    return np.array([np.log10(P[(fr>=lo)&(fr<hi)].mean()+1e-20)
                     if ((fr>=lo)&(fr<hi)).any() else -20.0 for lo,hi in zip(e[:-1],e[1:])])

def auc(idx, yv, grp):
    X = np.stack([bandpower(f['traces'][f'{i:04d}'][:], fs[i]) for i in idx])
    folds = min(5, len(set(grp.tolist())), int(min(np.bincount(yv))))
    if folds < 2: return None, X
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pr = cross_val_predict(pipe, X, yv, cv=GroupKFold(folds), groups=grp,
                           method='predict_proba')[:,1]
    return roc_auc_score(yv, pr), X

all_i = np.arange(len(fs))
dmot = all_i[(strain=='MG1655 dMotAB') & (np.round(fs,1)==1666.7)]
a, X = auc(dmot, treated[dmot].astype(int), unit[dmot])
print(f'dMotAB real contrast                       AUC = {a:.3f}')

# permutation null: shuffle the label, keep the grouping
nulls = []
grp = unit[dmot]; yv = treated[dmot].astype(int)
for _ in range(30):
    yp = rng.permutation(yv)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pr = cross_val_predict(pipe, X, yp, cv=GroupKFold(5), groups=grp, method='predict_proba')[:,1]
    nulls.append(roc_auc_score(yp, pr))
nulls = np.array(nulls)
print(f'  permuted-label null (30 draws)           mean {nulls.mean():.3f}  '
      f'95th pct {np.percentile(nulls,95):.3f}  max {nulls.max():.3f}')

# sham contrasts between groups that differ by session or strain, never by drug
def sham(ia, ib, tag):
    idx = np.concatenate([ia, ib]); yv = np.r_[np.zeros(len(ia),int), np.ones(len(ib),int)]
    a, _ = auc(idx, yv, unit[idx])
    print(f'  {tag:40s} AUC = ' + (f'{a:.3f}' if a else 'n/a'))

print('sham contrasts (no drug difference at all):')
u = ~treated
mt29_23 = all_i[u & (label=='DLG Flagellaless MTB29 R308_23')]
mt28_23 = all_i[u & (label=='DLG Flagellaless MTB28 R308_23')]
mt29_50 = all_i[u & (label=='DLG fliC flagellaless Ecoli R308_50')]
sham(mt29_23, mt28_23, 'MTB29 vs MTB28, same chip run')
sham(mt29_23, mt29_50, 'MTB29 run R308_23 vs run R308_50')
