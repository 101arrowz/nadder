export interface StringArray {
  [index: number]: string;
}
export class StringArray {
  // buffer
  private b: string[];
  // size
  readonly length: number;

  constructor(size: number);
  constructor(buf: string[]);
  constructor(sizeOrBuf: number | string[]) {
    if (typeof sizeOrBuf == 'number') {
      this.b = Array(sizeOrBuf);
      this.b.fill('');
    } else {
      this.b = sizeOrBuf;
      if (this.b.some(v => typeof v != 'string')) {
        throw new TypeError('string array must contain only strings');
      }
    }
    const size = this.length = this.b.length;
    return new Proxy(this, {
      get: (target, name) => {
        if (typeof name == 'symbol') return target[name];
        const ind = +name;
        if (name != 'NaN' && isNaN(ind)) return target[name];
        if (!Number.isInteger(ind) || ind < 0 || ind > size) return;
        return this.b[ind];
      },
      set: (target, name, value) => {
        if (typeof name != 'symbol') {
          const ind = +name;
          if (name == 'NaN' || !isNaN(ind)) {
            if (Number.isInteger(ind) && ind >= 0 && ind < size) {
              target.b[ind] = `${value}`;
            }
            return true;
          }
        }
        target[name] = value;
        return true;
      }
    });
  }

  toArray(): string[] {
    return this.b.slice();
  }
}