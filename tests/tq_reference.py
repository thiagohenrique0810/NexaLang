"""Independent Python TQ_MSE_SRHT V1 oracle with an explicit F32LE codebook.

No runtime codec, native library or Lloyd-Max generator is imported. Integer
SplitMix64/xoshiro256** fixes SRHT signs. Operations round at the documented
F32 boundaries, and TQ02 records store an LE norm and LSB-first indices.
"""
import math
import struct

TRANSFORM_ID = 'SRHT_XOSHIRO256SS_V1'


def f32(value):
    try:
        result = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError('TQ reference values must fit finite F32') from error
    if not math.isfinite(result):
        raise ValueError('TQ reference values must fit finite F32')
    return result


def _signs(dim, seed):
    mask = (1 << 64) - 1
    rotate = lambda value, shift: ((value << shift) | (value >> (64 - shift))) & mask
    state = []
    for _ in range(4):
        seed = (seed + 0x9E3779B97F4A7C15) & mask
        word = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & mask
        state.append(word ^ (word >> 31))
    result = []
    for _ in range(dim):
        result.append(1 if (rotate(state[1] * 5 & mask, 7) * 9 & mask) & 1 else -1)
        temporary = (state[1] << 17) & mask
        state[2] ^= state[0]
        state[3] ^= state[1]
        state[1] ^= state[2]
        state[0] ^= state[3]
        state[2] ^= temporary
        state[3] = rotate(state[3], 45)
    return result


def _hadamard(values):
    values = list(values)
    half = 1
    while half < len(values):
        for start in range(0, len(values), 2 * half):
            for index in range(start, start + half):
                left, right = values[index], values[index + half]
                values[index], values[index + half] = f32(left + right), f32(left - right)
        half *= 2
    return values


class TQReference:
    def __init__(self, dim, bits, seed, codebook_f32le):
        if type(dim) is not int or not 0 < dim <= 1 << 20 or dim & (dim - 1):
            raise ValueError('TQ dimension must be a power of two up to 2**20')
        if type(bits) is not int or not 1 <= bits <= 8:
            raise ValueError('TQ bits must be an integer from 1 through 8')
        if type(seed) is not int or not -(1 << 31) <= seed < 1 << 31:
            raise ValueError('TQ seed must be a signed 32-bit integer')
        if (not isinstance(codebook_f32le, str) or len(codebook_f32le) != 8 * (1 << bits)
                or any(char not in '0123456789abcdef' for char in codebook_f32le)):
            raise ValueError('TQ reference requires an explicit canonical F32LE codebook')
        self.centroids = struct.unpack('<' + 'f' * (1 << bits), bytes.fromhex(codebook_f32le))
        if any(not math.isfinite(value) for value in self.centroids):
            raise ValueError('TQ centroids must be finite')
        if any(left >= right for left, right in zip(self.centroids, self.centroids[1:])):
            raise ValueError('TQ centroids must strictly increase')
        self.boundaries = [f32(f32(left + right) * 0.5)
                           for left, right in zip(self.centroids, self.centroids[1:])]
        self.dim, self.bits, self.seed = dim, bits, seed
        self.codebook_f32le = codebook_f32le
        self.row_bytes = 8 + (dim * bits + 7) // 8
        self.signs = _signs(dim, seed)
        self.denominator = f32(math.sqrt(dim))
        self.inverse = f32(1 / self.denominator)

    def encode(self, values):
        values = [f32(value) for value in values]
        if len(values) != self.dim:
            raise ValueError('TQ reference input width mismatch')
        norm64 = math.sqrt(sum(value * value for value in values))
        norm = f32(norm64)
        indices = 0
        if norm:
            rotated = _hadamard([f32(value / norm64) * sign
                                 for value, sign in zip(values, self.signs)])
            rotated = [f32(value / self.denominator if self.dim < 4 else value * self.inverse)
                       for value in rotated]
            for index, value in enumerate(rotated):
                indices |= sum(value > boundary for boundary in self.boundaries) << (index * self.bits)
        return b'TQ02' + struct.pack('<f', norm) + indices.to_bytes(self.row_bytes - 8, 'little')

    def decode(self, data):
        data = bytes(data)
        if len(data) != self.row_bytes or data[:4] != b'TQ02':
            raise ValueError('TQ02 record header/size mismatch')
        norm_bits = int.from_bytes(data[4:8], 'little')
        if norm_bits & 0x80000000 or norm_bits & 0x7f800000 == 0x7f800000:
            raise ValueError('TQ02 norm must be finite and nonnegative, including positive zero')
        norm = struct.unpack_from('<f', data, 4)[0]
        indices = int.from_bytes(data[8:], 'little')
        if indices >> (self.dim * self.bits) or (not norm and indices):
            raise ValueError('TQ02 unused bits and zero-norm indices must be zero')
        if not norm:
            return [0.0] * self.dim
        values = [self.centroids[(indices >> (index * self.bits)) & ((1 << self.bits) - 1)]
                  for index in range(self.dim)]
        if self.dim < 4:
            values = [f32(value * f32(sign / self.denominator))
                      for value, sign in zip(_hadamard(values), self.signs)]
        else:
            values = _hadamard([f32(value * self.inverse) for value in values])
            values = [value * sign for value, sign in zip(values, self.signs)]
        return [f32(value * norm) for value in values]
