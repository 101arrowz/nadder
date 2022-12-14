import { DataType } from '../../../core/datatype';
import { Complex } from '../../types';
import { opImpl, ufunc } from '../ufunc';

const identity = <T>(x: T) => x;

const complexTypeImpl = (complexImpl: (a: Complex, b: Complex) => Complex) => {
  const fixedComplexImpl = (a: number | boolean | bigint | Complex, b: number | boolean | bigint | Complex) => {
    if (typeof a != 'object') a = { real: Number(a), imag: 0 };
    if (typeof b != 'object') b = { real: Number(b), imag: 0 };
    return complexImpl(a, b);
  }
  return opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex]] as const, [DataType.Complex] as const, fixedComplexImpl);
}

const intTypeImpls = (impl: (a: number, b: number) => number, bigImpl: (a: bigint, b: bigint) => bigint) => {
  const fixedBoolImpl = (a: boolean, b: boolean) => impl(a as unknown as number, b as unknown as number) as unknown as boolean;
  const fixedBigImpl = (a: number | boolean | bigint, b: number | boolean | bigint) => {
    if (typeof a == 'number' || typeof a == 'boolean') a = BigInt(a);
    if (typeof b == 'number' || typeof b == 'boolean') b = BigInt(b);
    return bigImpl(a, b);
  };
  return [
    opImpl([[DataType.Bool], [DataType.Bool]] as const, [DataType.Bool] as const, fixedBoolImpl),
    opImpl([[DataType.Bool, DataType.Uint8], [DataType.Bool, DataType.Uint8]] as const, [DataType.Uint8] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8], [DataType.Bool, DataType.Int8]] as const, [DataType.Int8] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16]] as const, [DataType.Uint16] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16]] as const, [DataType.Int16] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32]] as const, [DataType.Uint32] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32]] as const, [DataType.Int32] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32, DataType.Uint64], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32, DataType.Uint64]] as const, [DataType.Uint64] as const, fixedBigImpl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Int64], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Int64]] as const, [DataType.Int64] as const, fixedBigImpl)
  ] as const;
}

const typeImpls = (impl: (a: number, b: number) => number, bigImpl: (a: bigint, b: bigint) => bigint) => {
  const fixedSmallImpl = (a: number | boolean | bigint, b: number | boolean | bigint) => {
    if (typeof a == 'bigint') a = Number(a);
    if (typeof b == 'bigint') b = Number(b);
    return impl(a as number, b as number);
  };
  return [
    ...intTypeImpls(impl, bigImpl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32]] as const, [DataType.Float32] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64]] as const, [DataType.Float64] as const, fixedSmallImpl),
  ] as const;
}

export const add = ufunc(
  'add',
  2,
  1,
  0,
  ...typeImpls((a, b) => a + b, (a, b) => a + b),
  complexTypeImpl((a, b) => ({ real: a.real + b.real, imag: a.imag + b.imag }))
);

export const sub = ufunc(
  'sub',
  2,
  1,
  null,
  ...typeImpls((a, b) => a - b, (a, b) => a - b),
  complexTypeImpl((a, b) => ({ real: a.real - b.real, imag: a.imag - b.imag }))
);

export const mul = ufunc(
  'mul',
  2,
  1,
  1,
  ...typeImpls((a, b) => a * b, (a, b) => a * b),
  complexTypeImpl((a, b) => ({ real: a.real * b.real - a.imag * b.imag, imag: a.real * b.imag + a.imag * b.real }))
);

export const fdiv = ufunc(
  'fdiv',
  2,
  1,
  null,
  ...typeImpls((a, b) => Math.floor(a / b), (a, b) => a / b)
);

export const bitand = ufunc(
  'bitand',
  2,
  1,
  -1,
  ...intTypeImpls((a, b) => a & b, (a, b) => a & b)
);

export const bitor = ufunc(
  'bitor',
  2,
  1,
  0,
  ...intTypeImpls((a, b) => a | b, (a, b) => a | b)
);

export const bitxor = ufunc(
  'bitxor',
  2,
  1,
  0,
  ...intTypeImpls((a, b) => a ^ b, (a, b) => a ^ b)
);

export const shl = ufunc(
  'shl',
  2,
  1,
  null,
  ...intTypeImpls((a, b) => a << b, (a, b) => a << b)
);

export const shr = ufunc(
  'shr',
  2,
  1,
  null,
  ...intTypeImpls((a, b) => a >> b, (a, b) => a >> b)
);

const bnot = <T extends number | bigint>(a: T): T => ~a as T;

export const bitnot = ufunc(
  'bitnot',
  1,
  1,
  null,
  opImpl([[DataType.Bool, DataType.Int8]] as const, [DataType.Int8] as const, bnot as (a: number | boolean) => number),
  opImpl([[DataType.Uint8]] as const, [DataType.Uint8] as const, bnot),
  opImpl([[DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, bnot),
  opImpl([[DataType.Int16]] as const, [DataType.Int16] as const, bnot),
  opImpl([[DataType.Uint16]] as const, [DataType.Uint16] as const, bnot),
  opImpl([[DataType.Int32]] as const, [DataType.Int32] as const, bnot),
  opImpl([[DataType.Uint32]] as const, [DataType.Uint32] as const, bnot),
  opImpl([[DataType.Int64]] as const, [DataType.Int64] as const, bnot),
  opImpl([[DataType.Uint64]] as const, [DataType.Uint64] as const, bnot),
);

export const pow = ufunc(
  'pow',
  2,
  1,
  null,
  ...typeImpls((a, b) => a ** b, (a, b) => a ** b),
  complexTypeImpl((a, b) => {
    // TODO
    throw new TypeError('pow is not implemented for complex numbers');
  })
);

export const mod = ufunc(
  'mod',
  2,
  1,
  null,
  ...typeImpls((a, b) => a % b, (a, b) => a % b),
);

export const abs = ufunc(
  'abs',
  1,
  1,
  null,
  opImpl([[DataType.Bool]] as const, [DataType.Bool] as const, identity),
  opImpl([[DataType.Int8]] as const, [DataType.Int8] as const, Math.abs),
  opImpl([[DataType.Uint8]] as const, [DataType.Uint8] as const, Math.abs),
  opImpl([[DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, Math.abs),
  opImpl([[DataType.Int16]] as const, [DataType.Int16] as const, Math.abs),
  opImpl([[DataType.Uint16]] as const, [DataType.Uint16] as const, Math.abs),
  opImpl([[DataType.Int32]] as const, [DataType.Int32] as const, Math.abs),
  opImpl([[DataType.Uint32]] as const, [DataType.Uint32] as const, Math.abs),
  opImpl([[DataType.Int64]] as const, [DataType.Int64] as const, a => a < 0 ? -a : a),
  opImpl([[DataType.Uint64]] as const, [DataType.Uint64] as const, a => a < 0 ? -a : a),
  opImpl([[DataType.Float32]] as const, [DataType.Float32] as const, Math.abs),
  opImpl([[DataType.Float64]] as const, [DataType.Float64] as const, Math.abs),
  opImpl([[DataType.Complex]] as const, [DataType.Float64] as const, (a) => +a),
);

export const conj = ufunc(
  'conj',
  1,
  1,
  null,
  opImpl([[DataType.Bool, DataType.Int8]] as const, [DataType.Int8] as const, (a) => +a),
  opImpl([[DataType.Uint8]] as const, [DataType.Uint8] as const, identity),
  opImpl([[DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, identity),
  opImpl([[DataType.Int16]] as const, [DataType.Int16] as const, identity),
  opImpl([[DataType.Uint16]] as const, [DataType.Uint16] as const, identity),
  opImpl([[DataType.Int32]] as const, [DataType.Int32] as const, identity),
  opImpl([[DataType.Uint32]] as const, [DataType.Uint32] as const, identity),
  opImpl([[DataType.Int64]] as const, [DataType.Int64] as const, identity),
  opImpl([[DataType.Uint64]] as const, [DataType.Uint64] as const, identity),
  opImpl([[DataType.Float32]] as const, [DataType.Float32] as const, identity),
  opImpl([[DataType.Float64]] as const, [DataType.Float64] as const, identity),
  opImpl([[DataType.Complex]] as const, [DataType.Complex] as const, (a) => ({ real: a.real, imag: -a.imag })),
);

export const pos = ufunc(
  'pos',
  1,
  1,
  null,
  opImpl([[DataType.Int8]] as const, [DataType.Int8] as const, identity),
  opImpl([[DataType.Uint8]] as const, [DataType.Uint8] as const, identity),
  opImpl([[DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, identity),
  opImpl([[DataType.Int16]] as const, [DataType.Int16] as const, identity),
  opImpl([[DataType.Uint16]] as const, [DataType.Uint16] as const, identity),
  opImpl([[DataType.Int32]] as const, [DataType.Int32] as const, identity),
  opImpl([[DataType.Uint32]] as const, [DataType.Uint32] as const, identity),
  opImpl([[DataType.Int64]] as const, [DataType.Int64] as const, identity),
  opImpl([[DataType.Uint64]] as const, [DataType.Uint64] as const, identity),
  opImpl([[DataType.Float32]] as const, [DataType.Float32] as const, identity),
  opImpl([[DataType.Float64]] as const, [DataType.Float64] as const, identity),
  opImpl([[DataType.Complex]] as const, [DataType.Complex] as const, identity),
);

const negate = <T extends number | bigint>(a: T) => -a as T;

export const neg = ufunc(
  'neg',
  1,
  1,
  null,
  opImpl([[DataType.Int8]] as const, [DataType.Int8] as const, negate),
  opImpl([[DataType.Uint8]] as const, [DataType.Uint8] as const, negate),
  opImpl([[DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, negate),
  opImpl([[DataType.Int16]] as const, [DataType.Int16] as const, negate),
  opImpl([[DataType.Uint16]] as const, [DataType.Uint16] as const, negate),
  opImpl([[DataType.Int32]] as const, [DataType.Int32] as const, negate),
  opImpl([[DataType.Uint32]] as const, [DataType.Uint32] as const, negate),
  opImpl([[DataType.Int64]] as const, [DataType.Int64] as const, negate),
  opImpl([[DataType.Uint64]] as const, [DataType.Uint64] as const, negate),
  opImpl([[DataType.Float32]] as const, [DataType.Float32] as const, negate),
  opImpl([[DataType.Float64]] as const, [DataType.Float64] as const, negate),
  opImpl([[DataType.Complex]] as const, [DataType.Complex] as const, (a) => ({ real: -a.real, imag: -a.imag })),
);

export const div = ufunc(
  'div',
  2,
  1,
  null,
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32], [DataType.Float32]] as const, [DataType.Float32] as const, (a, b) => +a / b),
  opImpl([[DataType.Float32], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32]] as const, [DataType.Float32] as const, (a, b) => a / +b),
  opImpl([
    [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32, DataType.Int32, DataType.Uint32, DataType.Int64, DataType.Uint64, DataType.Float64],
    [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32, DataType.Int32, DataType.Uint32, DataType.Int64, DataType.Uint64, DataType.Float64]
  ] as const, [DataType.Float64] as const, (a, b) => Number(a) / Number(b)),
  complexTypeImpl((a, b) => {
    const denom = Math.hypot(b.real, b.imag);
    return { real: (a.real * b.real + a.imag * b.imag) / denom, imag: (a.imag * b.real - a.real * b.imag) / denom };
  })
);

const floatTypeImpls = (impl: (a: number) => number, complexImpl: (a: Complex) => Complex) => {
  const fixedSmallImpl = (a: number | boolean | bigint) => {
    if (typeof a == 'bigint') a = Number(a);
    return impl(a as number);
  };
  const fixedComplexImpl = (a: number | boolean | bigint | Complex) => {
    if (typeof a != 'object') a = { real: Number(a), imag: 0 };
    return complexImpl(a);
  }
  return [
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32]] as const, [DataType.Float32] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32, DataType.Int32, DataType.Uint32, DataType.Int64, DataType.Uint64, DataType.Float64]] as const, [DataType.Float64] as const, fixedSmallImpl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32, DataType.Int32, DataType.Uint32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex]] as const, [DataType.Complex] as const, fixedComplexImpl),
  ] as const;
};

const complexSqrt = (a: Complex) => {
  const l = Math.hypot(a.real, a.imag);
  return {
    real: Math.sqrt((l + a.real) / 2),
    imag: Math.sign(a.imag) * Math.sqrt((l - a.real) / 2)
  };
};

export const sqrt = ufunc(
  'sqrt',
  1,
  1,
  null,
  ...floatTypeImpls(Math.sqrt, complexSqrt)
);

export const exp = ufunc(
  'exp',
  1,
  1,
  null,
  ...floatTypeImpls(Math.exp, a => {
    const base = Math.exp(a.real);
    return { real: base * Math.cos(a.imag), imag: base * Math.sin(a.imag) };
  })
);

export const exp2 = ufunc(
  'exp2',
  1,
  1,
  null,
  ...floatTypeImpls(a => 2 ** a, a => {
    const base = 2 ** a.real;
    const rad = a.imag * Math.LN2;
    return { real: base * Math.cos(rad), imag: base * Math.sin(rad) };
  })
);

export const expm1 = ufunc(
  'expm1',
  1,
  1,
  null,
  ...floatTypeImpls(Math.expm1, a => {
    // This loses precision if a.real and a.imag are near 0 - possible TODO
    // Though expm1 probably shouldn't be used for complex numbers anyway
    return { real: Math.exp(a.real) * Math.cos(a.imag) - 1, imag: Math.exp(a.real) * Math.sin(a.imag) };
  })
);

export const sin = ufunc(
  'sin',
  1,
  1,
  null,
  ...floatTypeImpls(Math.sin, a => ({
    real: Math.sin(a.real) * Math.cosh(a.imag),
    imag: Math.cos(a.real) * Math.sinh(a.imag),
  }))
);

export const cos = ufunc(
  'cos',
  1,
  1,
  null,
  ...floatTypeImpls(Math.cos, a => ({
    real: Math.cos(a.real) * Math.cosh(a.imag),
    imag: -Math.sin(a.real) * Math.sinh(a.imag),
  }))
);

export const tan = ufunc(
  'tan',
  1,
  1,
  null,
  ...floatTypeImpls(Math.tan, a => {
    const denom = Math.cos(2 * a.real) + Math.cosh(2 * a.imag);
    return {
      real: Math.sin(2 * a.real) / denom,
      imag: Math.sinh(2 * a.imag) / denom,
    };
  })
);


const relopImpl = (impl: (a: boolean | number | bigint, b: boolean | number | bigint) => boolean) => 
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64]] as const, [DataType.Bool] as const, impl);

const complexRelopImpl = (impl: (a: Complex, b: Complex) => boolean) => 
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex]] as const, [DataType.Bool] as const, (a, b) => {
    if (typeof a != 'object') a = { real: Number(a), imag: 0 };
    if (typeof b != 'object') b = { real: Number(b), imag: 0 };
    return impl(a, b);
  });

export const gt = ufunc(
  'gt',
  2,
  1,
  null,
  relopImpl((a, b) => a > b)
);

export const gte = ufunc(
  'gte',
  2,
  1,
  null,
  relopImpl((a, b) => a >= b)
);

export const lt = ufunc(
  'gt',
  2,
  1,
  null,
  relopImpl((a, b) => a < b)
);

export const lte = ufunc(
  'gte',
  2,
  1,
  null,
  relopImpl((a, b) => a <= b)
);

export const and = ufunc(
  'and',
  2,
  1,
  true,
  relopImpl((a, b) => (a && b) as unknown as boolean),
  complexRelopImpl((a, b) => ((a.real || a.imag) && (b.real || b.imag)) as unknown as boolean)
);

export const or = ufunc(
  'or',
  2,
  1,
  false,
  relopImpl((a, b) => (a || b) as unknown as boolean),
  complexRelopImpl((a, b) => ((a.real || a.imag) || (b.real || b.imag)) as unknown as boolean)
);

export const xor = ufunc(
  'xor',
  2,
  1,
  false,
  relopImpl((a, b) => !a != !b),
  complexRelopImpl((a, b) => !a.real && !a.imag != !b.real && !b.imag)
);

export const not = ufunc(
  'not',
  1,
  1,
  null,
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64]] as const, [DataType.Bool] as const, a => !a),
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex]] as const, [DataType.Bool] as const, a => typeof a == 'object' ? !a.real && !a.imag : !a)
);