import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from _common import args
_a = args(needs_mat=True)
PATH = _a.mat
OUT = _a.out

import numpy as np, h5py
from matread.mcos import elements, read_tag, parse_matrix



raw = open(PATH,'rb').read()
b = list(elements(raw))[1][1]
dt, ln, data, nxt = read_tag(b, 0); p = data
for _ in range(3): _, _, _, p = read_tag(b, p)
dtv, nv, dv, _ = read_tag(b, p)
fw = parse_matrix(b[dv:dv+nv], 8, max_depth=30)['fields']['MCOS']['payload']['items']
col = dict(zip(fw[9]['items'], fw[4]['items']))
cs = lambda c: [x if isinstance(x, str) else '' for x in c['items']]

y      = col['y']['items']
t_end  = col['t_end'].ravel()
drum   = col['drum_label'].ravel().astype(int)
std_nm = col['std_nm'].ravel()
label  = cs(col['label']); anti = cs(col['antibiotic'])
conc   = cs(col['concentration']); strain = cs(col['strain'])
n = len(y)

# The strain column carries a drug name (A22) or a growth medium (LB) for 188
# rows. The experiment label names the real strain in both cases, so recover
# it from there and record that the row was repaired.
FIX = {'A22': 'E.coli 2179', 'LB': 'MTB28'}
strain_fixed, repaired = [], []
for s, l in zip(strain, label):
    if s in FIX:
        strain_fixed.append(FIX[s]); repaired.append(True)
    else:
        strain_fixed.append(s); repaired.append(False)

lens = np.array([t.size for t in y])
fs   = lens / t_end
treated = np.array([a not in ('', 'None', 'none') for a in anti])
# drum_label restarts at 1 in each experiment, so it is not a drum identity on
# its own. The physical resonator is (experiment, drum_label).
unit = np.array([f'{l}|{d}' for l, d in zip(label, drum)])

with h5py.File(OUT + '/nanomotion.h5', 'w') as f:
    f.attrs['source_doi'] = '10.4121/384b7cf5-94da-4275-a6d6-a868eca8fec2'
    f.attrs['licence'] = 'CC BY-NC 4.0'
    f.attrs['n_records'] = n
    g = f.create_group('traces')
    for i, t in enumerate(y):
        g.create_dataset(f'{i:04d}', data=np.asarray(t, np.float32).ravel(),
                         compression='gzip', compression_opts=4)
    vl = h5py.special_dtype(vlen=str)
    for name, arr, d in [('fs', fs, 'f8'), ('duration_s', t_end, 'f8'),
                         ('n_samples', lens, 'i8'), ('std_nm', std_nm, 'f8'),
                         ('drum_label', drum, 'i8'), ('treated', treated, '?'),
                         ('repaired_strain', np.array(repaired), '?')]:
        f.create_dataset(name, data=np.asarray(arr, d))
    for name, arr in [('experiment', label), ('antibiotic', anti),
                      ('concentration', conc), ('strain', strain_fixed),
                      ('strain_as_deposited', strain), ('unit', unit.tolist())]:
        f.create_dataset(name, data=np.array(arr, dtype=object), dtype=vl)
print('wrote nanomotion.h5')
np.savez(OUT + '/meta2.npz', fs=fs, lens=lens, t_end=t_end, drum=drum,
         std_nm=std_nm, anti=np.array(anti), strain=np.array(strain_fixed),
         strain_raw=np.array(strain), label=np.array(label), conc=np.array(conc),
         treated=treated, unit=unit, repaired=np.array(repaired))
print('repaired strain rows:', sum(repaired))
