import { DataType } from '../../../core/datatype';
import { Complex } from '../../types';
import { opImpl, ufunc } from '../ufunc';

const anyRelopImpls = (numImpl: (a: number | boolean, b: number | boolean) => boolean, complexImpl: (a: Complex, b: Complex) => boolean, impl: (a: unknown, b: unknown) => boolean) => [
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64]] as const, [DataType.Bool] as const, numImpl),
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex]] as const, [DataType.Bool] as const, (a, b) => {
    if (typeof a != 'object') a = { real: Number(a), imag: 0 };
    if (typeof b != 'object') b = { real: Number(b), imag: 0 };
    return complexImpl(a, b);
  }),
  opImpl([[DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex, DataType.String, DataType.Any], [DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64, DataType.Complex, DataType.String, DataType.Any]] as const, [DataType.Bool] as const, impl)
] as const;

export const ne = ufunc(
  'ne',
  2,
  1,
  null,
  ...anyRelopImpls((a, b) => a != b, (a, b) => a.real != b.real || a.imag != b.imag, (a, b) => a != b)
);

export const eq = ufunc(
  'eq',
  2,
  1,
  null,
  ...anyRelopImpls((a, b) => a == b, (a, b) => a.real == b.real && a.imag == b.imag, (a, b) => a == b)
);