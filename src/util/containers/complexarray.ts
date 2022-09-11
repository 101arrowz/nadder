import { Complex, wrap } from '../types/complex';

/**
 * An array of complex numbers represented as an interleaved buffer of real and imaginary parts
 */
export interface ComplexArray {
  [index: number]: Complex;
}
export class ComplexArray {
  /**
   * The interleaved buffer this complex array accesses
   */
  readonly buffer: Float32Array | Float64Array;
  
  /**
   * The number of elements in the array
   */
  readonly length: number;

  /**
   * Creates a complex array filled with zeros
   * @param size The number of complex numbers in the array
   * @param float Whether or not to use 32-bit floats instead of 64-bit doubles
   */
  constructor(size: number, float?: boolean);
  /**
   * Wraps an interleaved buffer of real and imaginary parts into a complex array
   * @param interleaved The interleaved buffer to wrap
   */
  constructor(interleaved: Float32Array | Float64Array);
  constructor(sizeOrBuf: number | Float32Array | Float64Array, float?: boolean) {
    if (typeof sizeOrBuf == 'number') {
      this.buffer = new (float ? Float32Array : Float64Array)(sizeOrBuf << 1);
    } else {
      this.buffer = sizeOrBuf;
      if (this.buffer.length & 1) {
        throw new TypeError('interleaved complex buffer must be even length');
      }
    }
    const size = this.length = this.buffer.length >> 1;
    return new Proxy(this, {
      get: (target, name) => {
        if (typeof name == 'symbol') return target[name];
        const ind = +name;
        if (name != 'NaN' && isNaN(ind)) return target[name];
        if (!Number.isInteger(ind) || ind < 0 || ind > size) return;
        return wrap({
          get real() {
            return target.buffer[ind << 1];
          },
          set real(val: number) {
            target.buffer[ind << 1] = val;
          },
          get imag() {
            return target.buffer[(ind << 1) + 1];
          },
          set imag(val: number) {
            target.buffer[(ind << 1) + 1] = val;
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
              target.buffer[ind << 1] = value.real;
              target.buffer[(ind << 1) + 1] = value.imag;
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
   * Converts the complex array into a standard Array object
   * @returns An array of the elements
   */
  toArray(): Complex[] {
    const out = new Array<Complex>(this.length);
    for (let i = 0; i < out.length; ++i) {
      out[i] = wrap({
        real: this.buffer[i << 1],
        imag: this.buffer[(i << 1) + 1]
      });
    }
    return out;
  }

  /**
   * Creates a complex array from a standard Array of complex numbers
   * @param src The array of complex numbers to build off of
   * @returns A complex array with the provided contents
   */
  static fromArray(src: Complex[]) {
    const bs = new ComplexArray(src.length);
    for (let i = 0; i < src.length; ++i) {
      bs.buffer[i << 1] = src[i].real;
      bs.buffer[(i << 1) + 1] = src[i].imag;
    }
    return bs;
  }
}