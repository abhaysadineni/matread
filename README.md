# matread

Read MATLAB files that SciPy cannot open, and a worked characterisation of
one dataset that nobody could open without a MATLAB licence.

## The problem

MATLAB saves a `table`, and every other MCOS class, as an opaque object whose
contents live in a subsystem stream at the end of the file.
`scipy.io.loadmat` returns a `MatlabOpaque` placeholder for it.
`pymatreader` emits a warning and returns nothing usable. Neither library is
at fault, since the format is undocumented, but the practical effect is that
a public dataset deposited this way needs a commercial licence to read.

`matread` walks the raw MAT version 5 element tree and recovers the contents
with NumPy alone.

```python
from matread import read_table

names, columns = read_table("allDat.mat")
print(names)
# ['y', 't_start', 't_end', 'date', 'label', 'antibiotic',
#  'concentration', 'strain', 'var_nm', 'std_nm', 'drum_label']
```

Two details in the format account for most of the difficulty, and both fail
silently rather than raising.

MATLAB pads every data element to an eight byte boundary **except**
`miCOMPRESSED`. A parser that pads uniformly lands mid-stream on the second
variable and inflates plausible nonsense.

An opaque element carries no dimensions and no name. In those two slots sit
three character arrays giving the type system, the variable name and the
class. Read them in the ordinary way and the type system name `MCOS` arrives
as array dimensions `(77, 67, 79, 83)`.

Both are pinned in `tests/test_mcos.py`.

## The dataset

The scripts characterise a deposit of bacterial nanomotion recordings:
1,332 optical measurements of live bacteria sitting on graphene drums,
each labelled with the antibiotic that colony received.

> doi:10.4121/384b7cf5-94da-4275-a6d6-a868eca8fec2 (234,325,033 bytes)
> CC BY-NC 4.0

**The recordings are not redistributed here.** Download `allDat.mat` from the
DOI above and point the scripts at it. The deposit is CC BY-NC 4.0, so
non-commercial use is permitted with attribution and commercial use requires
written permission from the depositing laboratory. Nothing in this repository
grants that permission, and the depositing laboratory has not been contacted.

### Running it

```sh
pip install numpy h5py scikit-learn
python3 scripts/01_export.py /path/to/allDat.mat -o build
python3 scripts/02_arms.py -o build
python3 scripts/03_sham_controls.py -o build
```

The export takes a few minutes, most of it spent inflating a 718 MB
subsystem stream. `02_arms.py` takes about fifteen minutes on one core.

### What the deposit actually contains

1,332 recordings, 538 treated and 794 untreated, over 902 distinct physical
resonators in 23 experiments. float32 for 874 of them, float64 for 458.
Displacement standard deviation runs 0.61 to 45.63 nm, median 4.60 nm.

The sampling rate is **not uniform** and the deposit does not state it. It is
recovered as sample count over recorded duration.

| Rate | Recordings | Duration | Samples |
|---|---|---|---|
| 3,333.33 Hz | 1,070 | 30 s | 100,000 |
| 1,666.67 Hz | 226 | 60 s | 100,000 |
| 1,068.42 Hz | 19 | 120 s | 128,200 |
| 833.33 Hz | 16 | 120 s | 100,000 |
| 213.68 Hz | 1 | 600 s | 128,200 |

Three label defects, all corrected in the export with the affected row
flagged in `repaired_strain`.

The `strain` column holds a drug name (`A22`, 108 rows) or a growth medium
(`LB`, 80 rows). The experiment label names the real strain in both cases, so
the value is recoverable. Trained on the column as deposited, a model treats
188 rows as belonging to strains that do not exist.

Ciprofloxacin appears as both `Ciprofloxacin` and `Ciprofloxacine`, splitting
one compound of 136 recordings into groups of 100 and 36.

The `date` column is present in the schema and was deposited empty, so
acquisition order is not recoverable and time cannot be excluded as a
confound.

`drum_label` restarts at 1 in each experiment, so it is not a resonator
identity on its own. The physical resonator is `(experiment, drum_label)`.

### The result, and the control that bounds it

Ten experiments carry both classes on shared drums. Summed across them, 383
drums hold both an untreated and a treated recording taken within the same
experiment, which is the only structure in this deposit that supports a
comparison.

Within each arm, folds split by drum, eight log band powers are compared
against the deposit's own amplitude summary (`std_nm`). Spectral shape wins
in eight of ten arms, and the result holds in a strain whose flagellar motor
has been deleted, so the signal is neither swimming nor loudness.

| Experiment | Rate | n | Drums | Amplitude | Spectrum | Null 95th |
|---|---|---|---|---|---|---|
| MG1655 dMotAB, ciprofloxacin | 1666.7 | 200 | 100 | 0.673 | 0.919 | 0.555 |
| Rifampicin | 3333.4 | 166 | 83 | 0.771 | 0.965 | 0.574 |
| E. coli 2179, mixed | 3333.4 | 101 | 34 | 0.709 | 0.837 | 0.538 |
| E. coli 6300, mixed | 3333.4 | 80 | 40 | 0.811 | 0.966 | 0.562 |
| Ciprofloxacin | 3333.4 | 72 | 36 | 0.674 | 1.000 | 0.584 |
| Cadaverine | 3333.4 | 68 | 34 | 0.883 | 0.771 | 0.580 |
| DNP | 3333.4 | 54 | 27 | 0.975 | 1.000 | 0.528 |
| Kanamycin, chloramphenicol | 3333.4 | 42 | 14 | 0.658 | 0.796 | 0.715 |
| Cadaverine, second run | 1068.4 | 20 | 10 | 0.740 | 0.900 | 0.693 |
| Bacillus | 833.3 | 10 | 5 | 0.720 | 0.360 | 0.880 |

Null columns come from permuting the labels twenty times and repeating the
whole procedure. Seven arms place the spectral result clearly above their own
95th percentile. The Bacillus arm has five drums and a null reaching 0.880,
so it supports no claim in either direction.

**Read this before using the data for anything.** Two contrasts were run in
which no drug difference exists at all, only a difference of strain or of
chip run, using untreated recordings on both sides.

| Contrast | AUC |
|---|---|
| MTB29 against MTB28, same chip run | 0.996 |
| MTB29 run R308_23 against run R308_50 | 0.997 |

The same eight features identify which chip run a recording came from almost
perfectly. The apparatus signature is larger than the biological effect. Any
split that puts one experiment in training and another in test will score
high for the wrong reason, and the full 1,332 recordings cannot simply be
pooled into a classification task. Every number above is therefore computed
inside a single experiment with folds assigned by physical drum.

## Licence

Code in this repository is MIT. The dataset it reads is not included and
carries its own CC BY-NC 4.0 terms, described above.
