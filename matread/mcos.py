"""Minimal MAT v5 element walker, enough to reach inside an MCOS table.

MATLAB saves a `table` as an opaque MCOS object. scipy and pymatreader both
stop at the opaque boundary, so a dataset deposited this way is unreadable
without MATLAB. This walks the raw element tree instead.
"""
from __future__ import annotations
import struct
import zlib
import numpy as np

MI = {1:'miINT8',2:'miUINT8',3:'miINT16',4:'miUINT16',5:'miINT32',6:'miUINT32',
      7:'miSINGLE',9:'miDOUBLE',12:'miINT64',13:'miUINT64',14:'miMATRIX',
      15:'miCOMPRESSED',16:'miUTF8',17:'miUTF16',18:'miUTF32'}
MX = {1:'cell',2:'struct',3:'object',4:'char',5:'sparse',6:'double',7:'single',
      8:'int8',9:'uint8',10:'int16',11:'uint16',12:'int32',13:'uint32',
      14:'int64',15:'uint64',17:'opaque'}
NP = {1:'<i1',2:'<u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',9:'<f8',
      12:'<i8',13:'<u8',16:'<u1',17:'<u2',18:'<u4'}


def read_tag(b: bytes, o: int):
    """Returns (dtype, nbytes, data_offset, next_offset).

    Data elements are padded to an 8-byte boundary, with one exception that
    costs an afternoon if you miss it: MATLAB does not pad miCOMPRESSED (15).
    Pad it anyway and every element after the first lands mid-stream, which
    reads as plausible-looking garbage rather than as an error.
    """
    raw, ln = struct.unpack_from('<II', b, o)
    if (raw >> 16) != 0:                      # small element: type in low half
        dt, n = raw & 0xFFFF, raw >> 16
        return dt, n, o + 4, o + 8
    pad = 0 if raw == 15 else (8 - ln % 8) % 8
    return raw, ln, o + 8, o + 8 + ln + pad


def elements(b: bytes, start: int = 128):
    """Yield (dtype, data, length) for each top-level element, inflating
    miCOMPRESSED transparently. `start` is 128 for a file, 0 for a blob."""
    o = start
    while o + 8 <= len(b):
        dt, ln, data, nxt = read_tag(b, o)
        if dt == 15:
            inner = zlib.decompress(b[data:data + ln])
            yield read_tag(inner, 0)[0], inner, len(inner)
        else:
            yield dt, b[o:nxt], ln
        o = nxt


def read_numeric(b, dt, n, off):
    fmt = NP.get(dt)
    if fmt is None:
        return None
    return np.frombuffer(b, fmt, n // np.dtype(fmt).itemsize, off)


def parse_matrix(b: bytes, o: int, depth: int = 0, max_depth: int = 12):
    """Parse one miMATRIX element at offset o. Returns a python object."""
    dt, ln, data, nxt = read_tag(b, o)
    if dt != 14:
        raise ValueError(f"not a miMATRIX at {o}, got {MI.get(dt, dt)}")
    if ln == 0:
        return None                      # MATLAB writes [] as a zero-length element
    end = data + ln
    p = data

    dt1, n1, d1, p = read_tag(b, p)                 # array flags
    flags = read_numeric(b, dt1, n1, d1)
    cls = int(flags[0]) & 0xFF
    is_logical = bool((int(flags[0]) >> 9) & 1)

    if cls == 17:
        # An opaque element does not carry dimensions or a name. It carries
        # three char arrays -- type system, variable name, class name -- and
        # then its payload. Reading those three slots as flags/dims/name is
        # what makes an MCOS file look like nonsense: the type system name
        # "MCOS" arrives as dimensions (77, 67, 79, 83).
        strs = []
        for _ in range(3):
            dts, ns, ds, p = read_tag(b, p)
            strs.append(b[ds:ds + ns].decode('latin-1'))
        payload = None
        if p < end:
            try:
                payload = parse_matrix(b, p, depth + 1, max_depth)
            except ValueError:
                dtp, np_, dp, _ = read_tag(b, p)
                payload = read_numeric(b, dtp, np_, dp)
        return {'_class': 'opaque', '_typesystem': strs[0], '_name': strs[1],
                '_classname': strs[2], 'payload': payload}

    dt2, n2, d2, p = read_tag(b, p)                 # dimensions
    dims = tuple(int(x) for x in read_numeric(b, dt2, n2, d2))

    dt3, n3, d3, p = read_tag(b, p)                 # name
    name = b[d3:d3 + n3].decode('latin-1')

    if depth > max_depth:
        return {'_class': MX.get(cls, cls), '_dims': dims, '_name': name,
                '_truncated': True}

    if cls == 1:                                     # cell
        items = []
        while p < end:
            try:
                items.append(parse_matrix(b, p, depth + 1, max_depth))
            except Exception as e:
                items.append({'_error': str(e)[:80]})
            _, _, _, p = read_tag(b, p)
        return {'_class': 'cell', '_dims': dims, '_name': name, 'items': items}

    if cls == 2:                                     # struct
        dtl, nl, dl, p = read_tag(b, p)              # field name length
        fnlen = int(read_numeric(b, dtl, nl, dl)[0])
        dtn, nn, dn, p = read_tag(b, p)              # field names
        nfields = nn // fnlen
        names = [b[dn + i * fnlen: dn + (i + 1) * fnlen].split(b'\x00')[0]
                 .decode('latin-1') for i in range(nfields)]
        out = {'_class': 'struct', '_dims': dims, '_name': name, 'fields': {}}
        for fn in names:
            try:
                out['fields'][fn] = parse_matrix(b, p, depth + 1, max_depth)
            except Exception as e:
                out['fields'][fn] = {'_error': str(e)[:80]}; break
            _, _, _, p = read_tag(b, p)
        return out

    if cls == 4:                                     # char
        dtc, nc, dc, _ = read_tag(b, p)
        arr = read_numeric(b, dtc, nc, dc)
        if arr is None:
            return ''
        return ''.join(chr(int(x)) for x in arr)

    # numeric / logical
    dtv, nv, dv, _ = read_tag(b, p)
    arr = read_numeric(b, dtv, nv, dv)
    if arr is None:
        return {'_class': MX.get(cls, cls), '_dims': dims, '_raw_dtype': dtv}
    try:
        arr = arr.reshape(dims, order='F')
    except Exception:
        pass
    return arr


def summarise(obj, indent=0, maxitems=6):
    pad = '  ' * indent
    if isinstance(obj, dict) and obj.get('_class') == 'struct':
        print(f"{pad}struct {obj['_dims']} fields={list(obj['fields'])}")
        for k, v in list(obj['fields'].items())[:maxitems]:
            print(f"{pad}  .{k}:")
            summarise(v, indent + 2, maxitems)
    elif isinstance(obj, dict) and obj.get('_class') == 'cell':
        print(f"{pad}cell {obj['_dims']} n={len(obj['items'])}")
        for v in obj['items'][:maxitems]:
            summarise(v, indent + 1, maxitems)
    elif isinstance(obj, np.ndarray):
        print(f"{pad}array {obj.shape} {obj.dtype}"
              + (f" first={obj.ravel()[:4]}" if obj.size else ""))
    elif isinstance(obj, str):
        print(f"{pad}char {obj[:60]!r}")
    else:
        print(f"{pad}{type(obj).__name__} {str(obj)[:100]}")
