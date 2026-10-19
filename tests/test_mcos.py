"""The two bugs that made this file unreadable, pinned.

Both were silent. Neither raised. Each returned data that looked plausible
until it was counted, which is the failure mode worth a regression test.
"""
import struct
import zlib

import numpy as np
import pytest

from matread.mcos import elements, parse_matrix, read_tag


def tag(dtype, payload, pad=True):
    out = struct.pack('<II', dtype, len(payload)) + payload
    if pad:
        out += b'\x00' * ((8 - len(payload) % 8) % 8)
    return out


class TestCompressedElementPadding:
    """MATLAB pads every element type to 8 bytes except miCOMPRESSED."""

    def test_compressed_element_is_not_padded(self):
        # 5 bytes of payload. A padding parser would skip 3 bytes too many.
        body = b'\x01\x02\x03\x04\x05'
        comp = zlib.compress(body)
        blob = struct.pack('<II', 15, len(comp)) + comp
        dt, ln, data, nxt = read_tag(blob, 0)
        assert dt == 15
        assert nxt == 8 + len(comp), 'compressed element must not be padded'

    def test_uncompressed_element_is_padded(self):
        blob = tag(2, b'\x01\x02\x03\x04\x05')
        dt, ln, data, nxt = read_tag(blob, 0)
        assert ln == 5
        assert nxt == 16, 'a 5-byte payload pads to the 8-byte boundary'

    def test_two_compressed_elements_both_reachable(self):
        # The regression itself: with uniform padding the second element
        # starts mid-stream and inflates to garbage or raises.
        parts = []
        for payload in (b'abcde', b'fghijkl'):
            c = zlib.compress(tag(2, payload))
            parts.append(struct.pack('<II', 15, len(c)) + c)
        stream = b'\x00' * 128 + b''.join(parts)
        got = [b for _, b, _ in elements(stream)]
        assert len(got) == 2, 'both compressed elements must be found'


class TestOpaqueElement:
    """An opaque element carries three char arrays, not flags/dims/name."""

    def build(self, typesystem, name, classname, payload_el):
        inner = (tag(6, struct.pack('<II', 17, 0))
                 + tag(1, typesystem.encode())
                 + tag(1, name.encode())
                 + tag(1, classname.encode())
                 + payload_el)
        return tag(14, inner)

    def test_reads_class_and_type_system(self):
        payload = tag(14, tag(6, struct.pack('<II', 6, 0))
                      + tag(5, struct.pack('<2i', 1, 2))
                      + tag(1, b'')
                      + tag(9, struct.pack('<2d', 1.5, 2.5)))
        blob = self.build('MCOS', 'allDat', 'table', payload)
        got = parse_matrix(blob, 0)
        assert got['_class'] == 'opaque'
        assert got['_typesystem'] == 'MCOS'
        assert got['_classname'] == 'table'

    def test_type_system_name_is_not_mistaken_for_dimensions(self):
        # The original symptom: "MCOS" read as dims (77, 67, 79, 83).
        payload = tag(14, tag(6, struct.pack('<II', 6, 0))
                      + tag(5, struct.pack('<2i', 1, 1))
                      + tag(1, b'')
                      + tag(9, struct.pack('<d', 1.0)))
        got = parse_matrix(self.build('MCOS', 'x', 'table', payload), 0)
        assert '_dims' not in got
        assert got['_typesystem'] == 'MCOS'

    def test_payload_is_parsed(self):
        payload = tag(14, tag(6, struct.pack('<II', 6, 0))
                      + tag(5, struct.pack('<2i', 1, 3))
                      + tag(1, b'')
                      + tag(9, struct.pack('<3d', 4.0, 5.0, 6.0)))
        got = parse_matrix(self.build('MCOS', 'x', 'table', payload), 0)
        assert np.allclose(np.asarray(got['payload']).ravel(), [4.0, 5.0, 6.0])
