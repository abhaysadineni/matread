"""Read MATLAB files that SciPy cannot open.

MATLAB saves a `table` (and any other MCOS class) as an opaque object. Both
`scipy.io.loadmat` and `pymatreader` stop at that boundary, so a dataset
deposited in that format needs a MATLAB licence to read. This package walks
the raw MAT v5 element tree instead and recovers the contents with NumPy.
"""
from matread.mcos import elements, parse_matrix, read_tag, summarise

__all__ = ["elements", "parse_matrix", "read_tag", "summarise", "read_table"]
__version__ = "0.1.0"


def read_table(path):
    """Return (column_names, columns) for a MAT file holding one MCOS table.

    columns is a list parallel to column_names. Numeric columns come back as
    NumPy arrays; cell columns as lists.
    """
    raw = open(path, "rb").read()
    els = list(elements(raw))
    if len(els) < 2:
        raise ValueError("no MCOS subsystem found; this file holds no table")
    b = els[1][1]
    _, _, data, _ = read_tag(b, 0)
    p = data
    for _ in range(3):
        _, _, _, p = read_tag(b, p)
    dtv, nv, dv, _ = read_tag(b, p)
    fw = parse_matrix(b[dv:dv + nv], 8, max_depth=30)["fields"]["MCOS"]["payload"]["items"]
    names = list(fw[9]["items"])
    cols = []
    for c in fw[4]["items"]:
        if isinstance(c, dict) and c.get("_class") == "cell":
            cols.append(list(c["items"]))
        else:
            cols.append(c)
    return names, cols
