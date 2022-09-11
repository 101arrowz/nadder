/**
 * An array that coerces all entries to strings
 */
export interface StringArray {
  [index: number]: string;
}
export class StringArray {
  /**
   * The buffer this string array accesses
   */
  readonly buffer: string[];

  /**
   * The number of elements in the array
   */
  readonly length: number;

  /**
   * Creates an array of empty strings with the given size
   * @param size The number of elements in the array
   */
  constructor(size: number);
  /**
   * Creates a string array from an Array object containing strings
   * @param buf The buffer to use
   */
  constructor(buf: string[]);
  constructor(sizeOrBuf: number | string[]) {
    if (typeof sizeOrBuf == 'number') {
      this.buffer = Array(sizeOrBuf);
      this.buffer.fill('');
    } else {
      this.buffer = sizeOrBuf;
      if (this.buffer.some(v => typeof v != 'string')) {
        throw new TypeError('string array must contain only strings');
      }
    }
    const size = this.length = this.buffer.length;
    return new Proxy(this, {
      get: (target, name) => {
        if (typeof name == 'symbol') return target[name];
        const ind = +name;
        if (name != 'NaN' && isNaN(ind)) return target[name];
        if (!Number.isInteger(ind) || ind < 0 || ind > size) return;
        return this.buffer[ind];
      },
      set: (target, name, value) => {
        if (typeof name != 'symbol') {
          const ind = +name;
          if (name == 'NaN' || !isNaN(ind)) {
            if (Number.isInteger(ind) && ind >= 0 && ind < size) {
              target.buffer[ind] = `${value}`;
            }
            return true;
          }
        }
        target[name] = value;
        return true;
      }
    });
  }
}