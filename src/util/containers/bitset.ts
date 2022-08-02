export interface Bitset {
  [index: number]: boolean;
}
export class Bitset {
  // buffer
  private b: Uint8Array;
  // size in bits
  readonly length: number;

  constructor(size: number) {
    this.b = new Uint8Array((size + 7) >> 3);
    this.length = size;
    return new Proxy(this, {
      get: (target, name) => {
        if (typeof name == 'symbol') return target[name];
        const ind = +name;
        if (name != 'NaN' && isNaN(ind)) return target[name];
        if (!Number.isInteger(ind) || ind < 0 || ind > size) return;
        return (target.b[ind >> 3] & (1 << (ind & 7))) > 0;
      },
      set: (target, name, value) => {
        if (typeof name != 'symbol') {
          const ind = +name;
          if (name == 'NaN' || !isNaN(ind)) {
            if (Number.isInteger(ind) && ind >= 0 && ind < size) {
              const bit = (1 << (ind & 7));
              if (value) target.b[ind >> 3] |= bit;
              else target.b[ind >> 3] &= ~bit;
            }
            return true;
          }
        }
        target[name] = value;
        return true;
      }
    });
  }

  toArray(): boolean[] {
    const out = new Array<boolean>(this.b.length << 3);
    for (let i = 0; i < this.b.length; ++i) {
      const ind = i << 3;

      out[ind] = (this.b[i] & 1) > 0;
      out[ind + 1] = (this.b[i] & 2) > 0;
      out[ind + 2] = (this.b[i] & 4) > 0;
      out[ind + 3] = (this.b[i] & 8) > 0;
      out[ind + 4] = (this.b[i] & 16) > 0;
      out[ind + 5] = (this.b[i] & 32) > 0;
      out[ind + 6] = (this.b[i] & 64) > 0;
      out[ind + 7] = (this.b[i] & 128) > 0;
    }
    out.length = this.length;
    return out;
  }

  static fromArray(src: boolean[]) {
    const bs = new Bitset(src.length);
    for (let i = 0; i < bs.b.length; ++i) {
      const ind = i << 3;
      bs.b[i] = (+src[ind + 7] << 7) | (+src[ind + 6] << 6) | (+src[ind + 5] << 5) |
                (+src[ind + 4] << 4)| (+src[ind + 3] << 3) | (+src[ind + 2] << 2) |
                (+src[ind + 1] << 6) | +src[ind];
    }
    return bs;
  }
}