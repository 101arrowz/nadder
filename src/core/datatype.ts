import { Bitset, ComplexArray } from '../util';

export enum DataType {
  Int8,
  Uint8,
  Uint8Clamped,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Float32,
  Float64,
  Complex,
  Bool,
  Int64,
  Uint64,
  Object
}

export type IntType = 
  | DataType.Int8
  | DataType.Uint8
  | DataType.Uint8Clamped
  | DataType.Int16
  | DataType.Uint16
  | DataType.Int32
  | DataType.Uint32;

export type NumericType = 
  | IntType
  | DataType.Float32
  | DataType.Float64
  | DataType.Complex;

export type BigNumericType = DataType.Int64 | DataType.Uint64;

export type AssignableType<T extends DataType> = T extends NumericType
  ? NumericType | DataType.Bool
  : T extends DataType.Object
    ? DataType
    : T extends DataType.Bool
      ? DataType.Bool
      : T extends BigNumericType
        ? BigNumericType
        : never;

export function isAssignable<T1 extends DataType>(dst: T1, src: DataType): src is AssignableType<T1> {
  if (dst == src || dst == DataType.Object) return true;
  if (dst == DataType.Bool) return src == DataType.Bool;
  if (dst == DataType.Int64 || dst == DataType.Uint64) return src == DataType.Int64 || src == DataType.Uint64;
  if (dst <= DataType.Float64) return src <= DataType.Bool;
  return false;
}

export const dataTypeBufferMap = {
  [DataType.Int8]: Int8Array,
  [DataType.Uint8]: Uint8Array,
  [DataType.Uint8Clamped]: Uint8ClampedArray,
  [DataType.Int16]: Int16Array,
  [DataType.Uint16]: Uint16Array,
  [DataType.Int32]: Int32Array,
  [DataType.Uint32]: Uint32Array,
  [DataType.Int64]: BigInt64Array,
  [DataType.Uint64]: BigUint64Array,
  [DataType.Float32]: Float32Array,
  [DataType.Float64]: Float64Array,
  [DataType.Complex]: ComplexArray,
  [DataType.Bool]: Bitset,
  [DataType.Object]: Array as { new(length: number): unknown[] }
};

export type DataTypeBuffer<T extends DataType> = InstanceType<(typeof dataTypeBufferMap)[T]>
export type IndexType<T extends DataType> = DataTypeBuffer<T>[number];