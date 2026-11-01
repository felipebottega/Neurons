"""Read a Godot 4 store_var() .neuronsim file and export it to JSON.
Usage: python read_neuronsim.py simulation.neuronsim [output.json]
"""
import json
import struct
import sys
from pathlib import Path

class GodotVariantReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def raw(self, count):
        chunk = self.data[self.pos:self.pos + count]
        if len(chunk) != count:
            raise ValueError(f"Unexpected end of file at offset {self.pos}")
        self.pos += count
        return chunk

    def u32(self): return struct.unpack('<I', self.raw(4))[0]
    def i32(self): return struct.unpack('<i', self.raw(4))[0]
    def i64(self): return struct.unpack('<q', self.raw(8))[0]
    def f32(self): return struct.unpack('<f', self.raw(4))[0]
    def f64(self): return struct.unpack('<d', self.raw(8))[0]

    def string(self):
        count = self.u32()
        s = self.raw(count).decode('utf-8')
        self.raw((-count) % 4)
        return s

    def values(self, n, method): return [method() for _ in range(n)]

    def variant(self):
        header = self.u32()
        kind = header & 0xffff
        flag = bool(header & 0x10000)
        if kind == 0: return None
        if kind == 1: return bool(self.u32())
        if kind == 2: return self.i64() if flag else self.i32()
        if kind == 3: return self.f64() if flag else self.f32()
        if kind in (4, 21, 22): return self.string()
        if kind == 5: return {'_type': 'Vector2', 'values': self.values(2, self.f32)}
        if kind == 6: return {'_type': 'Vector2i', 'values': self.values(2, self.i32)}
        if kind == 7: return {'_type': 'Rect2', 'values': self.values(4, self.f32)}
        if kind == 8: return {'_type': 'Rect2i', 'values': self.values(4, self.i32)}
        if kind == 9: return {'_type': 'Vector3', 'values': self.values(3, self.f32)}
        if kind == 10: return {'_type': 'Vector3i', 'values': self.values(3, self.i32)}
        if kind == 12: return {'_type': 'Vector4', 'values': self.values(4, self.f32)}
        if kind == 13: return {'_type': 'Vector4i', 'values': self.values(4, self.i32)}
        if kind == 20: return {'_type': 'Color', 'values': self.values(4, self.f32)}
        if kind == 27:
            for shift in (16, 18):
                typed = (header >> shift) & 3
                if typed == 1: self.u32()
                elif typed in (2, 3): self.string()
            count = self.u32() & 0x7fffffff
            result = {}
            for _ in range(count):
                key, value = self.variant(), self.variant()
                result[str(key)] = value
            return result
        if kind == 28:
            typed = (header >> 16) & 3
            if typed == 1: self.u32()
            elif typed in (2, 3): self.string()
            count = self.u32() & 0x7fffffff
            return self.values(count, self.variant)
        if kind == 29:
            count = self.u32()
            value = list(self.raw(count))
            self.raw((-count) % 4)
            return value
        packed = {30: (self.i32, 4), 31: (self.i64, 8), 32: (self.f32, 4), 33: (self.f64, 8), 35: (lambda: self.values(2, self.f32), 8), 36: (lambda: self.values(3, self.f32), 12), 37: (lambda: self.values(4, self.f32), 16), 38: (lambda: self.values(4, self.f32), 16)}
        if kind == 34: return self.values(self.u32(), self.string)
        if kind in packed:
            count = self.u32()
            return self.values(count, packed[kind][0])
        raise ValueError(f'Unsupported Godot Variant type {kind} at offset {self.pos - 4}')


def load(path):
    raw = Path(path).read_bytes()
    size = struct.unpack_from('<I', raw)[0]
    if size != len(raw) - 4:
        raise ValueError(f'Invalid length prefix: {size}; actual payload: {len(raw) - 4}')
    reader = GodotVariantReader(raw[4:])
    data = reader.variant()
    if reader.pos != len(reader.data):
        raise ValueError(f'Unread bytes: {len(reader.data) - reader.pos}')
    return data

if __name__ == '__main__':
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('simulation.neuronsim')
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.json')
    data = load(src)
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding='utf-8')
    print('Saved:', dst, '| Top-level fields:', list(data))
