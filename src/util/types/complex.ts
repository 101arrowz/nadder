/**
 * A complex number
 */
export interface Complex {
  /** Real part of the complex number */
  real: number;
  /** Imaginary part of the complex number */
  imag: number;
};

export function wrap(obj: Complex) {
  Object.defineProperty(obj, Symbol.toPrimitive, {
    value: function(hint: 'string' | 'number') {
      if (hint == 'number') return Math.hypot(this.real, this.imag);
      return this.toString();
    }
  });
  Object.defineProperty(obj, Symbol.for('nodejs.util.inspect.custom'), {
    value: function() {
      return this.toString();
    }
  });
  Object.defineProperty(obj, 'toString', {
    value: function() {
      const realPost = Number.isInteger(this.real) ? '.' : '';
      const imagPost = Number.isInteger(this.imag) ? '.' : '';
      return `${this.real + realPost}${this.imag < 0 ? '-' : '+'}${Math.abs(this.imag) + imagPost}i`;
    },
  });
  return obj;
}