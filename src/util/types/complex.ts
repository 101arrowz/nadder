export interface Complex {
  real: number;
  imag: number;
};

export function wrap(obj: Complex) {
  Object.defineProperty(obj, Symbol.toPrimitive, {
    value: function(hint: 'string' | 'number') {
      if (hint == 'number') return Math.hypot(this.real, this.imag);
      return this.toString();
    }
  });
  Object.defineProperty(obj, 'toString', {
    value: function() {
      return `${this.real} ${this.imag < 0 ? '-' : '+'} ${this.imag}i`;
    },
  });
  return obj;
}