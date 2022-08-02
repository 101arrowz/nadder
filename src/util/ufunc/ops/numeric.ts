import { DataType } from '../../../core/datatype';
import { Complex } from '../../types';
import { opImpl, ufunc } from '../ufunc';

const typeImpls = (impl: (a: number, b: number) => number, bigImpl: (a: bigint, b: bigint) => bigint, complexImpl: (a: Complex, b: Complex) => Complex) => {
  const fixedComplexImpl = (a: number | boolean | bigint | Complex, b: number | boolean | bigint | Complex) => {
    if (typeof a != 'object') a = { real: Number(a), imag: 0 };
    if (typeof b != 'object') b = { real: Number(b), imag: 0 };
    return complexImpl(a, b);
  }
  const fixedSmallImpl = (a: number | boolean | bigint, b: number | boolean | bigint) => {
    if (typeof a == 'bigint') a = Number(a);
    if (typeof b == 'bigint') b = Number(b);
    return impl(a as number, b as number);
  };
  const fixedBigImpl = (a: number | boolean | bigint, b: number | boolean | bigint) => {
    if (typeof a == 'number' || typeof a == 'boolean') a = BigInt(a);
    if (typeof b == 'number' || typeof b == 'boolean') b = BigInt(b);
    return bigImpl(a, b);
  };
  return [
    opImpl([[DataType.Bool, DataType.Uint8], [DataType.Bool, DataType.Uint8]] as const, [DataType.Uint8] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped]] as const, [DataType.Uint8Clamped] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8], [DataType.Bool, DataType.Int8]] as const, [DataType.Int8] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16]] as const, [DataType.Uint16] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16]] as const, [DataType.Int16] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32]] as const, [DataType.Uint32] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32]] as const, [DataType.Int32] as const, impl),
    opImpl([[DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32, DataType.Uint64], [DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32, DataType.Uint64]] as const, [DataType.Uint64] as const, fixedBigImpl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Int64], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Int64]] as const, [DataType.Int64] as const, fixedBigImpl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32]] as const, [DataType.Float32] as const, impl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64]] as const, [DataType.Float64] as const, fixedSmallImpl),
    opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex]] as const, [DataType.Complex] as const, fixedComplexImpl),
  ] as const;
}

export const add = ufunc(
  'add',
  2,
  1,
  ...typeImpls((a, b) => a + b, (a, b) => a + b, (a, b) => ({ real: a.real + b.real, imag: a.imag + b.imag })),
);
