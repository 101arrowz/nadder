export interface Complex {
  re: number;
  im: number;
};

export function wrap(obj: Complex) {
  Object.defineProperty(obj, Symbol.toPrimitive, {
    value: function(hint: 'string' | 'number') {
      if (hint == 'number') return Math.hypot(this.re, this.im);
      return this.toString();
    }
  });
  Object.defineProperty(obj, 'toString', {
    value: function() {
      return `${this.re} ${this.im < 0 ? '-' : '+'} ${this.im}i`;
    },
  });
  return obj;
}