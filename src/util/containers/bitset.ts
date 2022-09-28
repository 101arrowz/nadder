/**
 * An efficient boolean array using a byte array as a bitset
 */
export interface Bitset {
  [index: number]: boolean;
}
export class Bitset {
  /**
   * The byte buffer backing this bitset
   */
  readonly buffer: Uint8Array;

  /**
   * The number of elements in the bitset
   */
  readonly length: number;

  /**
   * The number of bits to skip from the start of the buffer
   */
  readonly offset: number;

  /**
   * Creates a new bitset for boolean ndarrays
   * @param size The length of the bitset to create
   * @returns A bitset with the provided buffer, or a new one if none was provided
   */
  constructor(size: number);
  /**
   * Creates a bitset from an existing buffer
   * @param src The buffer to use as the bitset backend
   * @param size The size of the bitset
   * @param offset The bit offset of the start of the bitset within the buffer
   */
  constructor(src: Uint8Array, size?: number, offset?: number);
  constructor(srcOrSize: number | Uint8Array, size?: number, offset?: number) {
    if (typeof srcOrSize == 'number') {
      this.buffer = new Uint8Array((srcOrSize + 7) >> 3);
      size = srcOrSize;
      this.offset = offset = 0;
    } else {
      if (!offset) offset = 0;
      if (size + offset > srcOrSize.length << 3) {
        throw new TypeError('bitset buffer too small');
      }
      this.buffer = srcOrSize;
      this.offset = offset;
    }
    this.length = size;
    return new Proxy(this, {
      get: (target, name) => {
        if (typeof name == 'symbol') return target[name];
        const ind = +name;
        if (name != 'NaN' && isNaN(ind)) return target[name];
        if (!Number.isInteger(ind) || ind < 0 || ind > size) return;
        return (target.buffer[(ind + offset) >> 3] & (1 << (ind & 7))) > 0;
      },
      set: (target, name, value) => {
        if (typeof name != 'symbol') {
          const ind = +name;
          if (name == 'NaN' || !isNaN(ind)) {
            if (Number.isInteger(ind) && ind >= 0 && ind < size) {
              const bit = (1 << (ind & 7));
              if (value) target.buffer[(ind + offset) >> 3] |= bit;
              else target.buffer[(ind + offset) >> 3] &= ~bit;
            }
            return true;
          }
        }
        target[name] = value;
        return true;
      }
    });
  }

  /**
   * Converts the bitset into a standard Array object
   * @returns An array of the elements
   */
  toArray(): boolean[] {
    const out = new Array<boolean>(this.buffer.length << 3);
    for (let i = 0; i < this.buffer.length; ++i) {
      const ind = i << 3;

      out[ind] = (this.buffer[i] & 1) > 0;
      out[ind + 1] = (this.buffer[i] & 2) > 0;
      out[ind + 2] = (this.buffer[i] & 4) > 0;
      out[ind + 3] = (this.buffer[i] & 8) > 0;
      out[ind + 4] = (this.buffer[i] & 16) > 0;
      out[ind + 5] = (this.buffer[i] & 32) > 0;
      out[ind + 6] = (this.buffer[i] & 64) > 0;
      out[ind + 7] = (this.buffer[i] & 128) > 0;
    }
    return out.slice(this.offset, this.offset + this.length);
  }

  /**
   * Creates a bitset from a standard Array of booleans
   * @param src The array of booleans to build off of
   * @returns A bitset with the provided contents
   */
  static fromArray(src: boolean[]) {
    const bs = new Bitset(src.length);
    for (let i = 0; i < bs.buffer.length; ++i) {
      const ind = i << 3;
      bs.buffer[i] = (+src[ind + 7] << 7) | (+src[ind + 6] << 6) | (+src[ind + 5] << 5) |
                     (+src[ind + 4] << 4)| (+src[ind + 3] << 3) | (+src[ind + 2] << 2) |
                     (+src[ind + 1] << 6) | +src[ind];
    }
    return bs;
  }
}