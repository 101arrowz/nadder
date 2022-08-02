import { Complex, wrap } from '../types/complex';

export interface ComplexArray {
  [index: number]: Complex;
}
export class ComplexArray {
  // buffer
  private b: Float32Array | Float64Array;
  // size
  readonly length: number;

  constructor(size: number, float?: boolean);
  constructor(interleaved: Float32Array | Float64Array);
  constructor(sizeOrBuf: number | Float32Array | Float64Array, float?: boolean) {
    if (typeof sizeOrBuf == 'number') {
      this.b = new (float ? Float32Array : Float64Array)(sizeOrBuf << 1);
    } else {
      this.b = sizeOrBuf;
      if (this.b.length & 1) {
        throw new TypeError('interleaved complex buffer must be even length');
      }
    }
    const size = this.length = this.b.length >> 1;
    return new Proxy(this, {
      get: (target, name) => {
        if (typeof name == 'symbol') return target[name];
        const ind = +name;
        if (name != 'NaN' && isNaN(ind)) return target[name];
        if (!Number.isInteger(ind) || ind < 0 || ind > size) return;
        return wrap({
          get real() {
            return target.b[ind << 1];
          },
          set real(val: number) {
            target.b[ind << 1] = val;
          },
          get imag() {
            return target.b[(ind << 1) + 1];
          },
          set imag(val: number) {
            target.b[(ind << 1) + 1] = val;
          }
        });
      },
      set: (target, name, value) => {
        if (typeof name != 'symbol') {
          const ind = +name;
          if (name == 'NaN' || !isNaN(ind)) {
            if (Number.isInteger(ind) && ind >= 0 && ind < size) {
              if (!value || typeof value.real != 'number' || typeof value.imag != 'number') {
                const real = +value;
                if (isNaN(real)) {
                  throw new TypeError('only complex and real numbers can be added to a complex array');
                }
                value = { real, imag: 0 };
              }
              target.b[ind << 1] = value.real;
              target.b[(ind << 1) + 1] = value.imag;
            }
            return true;
          }
        }
        target[name] = value;
        return true;
      }
    });
  }

  // non-reactive get
  get(ind: number) {
    if (!Number.isInteger(ind) || ind < 0 || ind > this.length) {
      throw new RangeError(`index ${ind} out of range for complex array of length ${this.length}`);
    }
    return wrap({
      real: this.b[ind << 1],
      imag: this.b[(ind << 1) + 1]
    });
  }

  toArray(): Complex[] {
    const out = new Array<Complex>(this.length);
    for (let i = 0; i < out.length; ++i) {
      out[i] = wrap({
        real: this.b[i << 1],
        imag: this.b[(i << 1) + 1]
      });
    }
    return out;
  }

  static fromArray(src: Complex[]) {
    const bs = new ComplexArray(src.length);
    for (let i = 0; i < src.length; ++i) {
      bs.b[i << 1] = src[i].real;
      bs.b[(i << 1) + 1] = src[i].imag;
    }
    return bs;
  }
}