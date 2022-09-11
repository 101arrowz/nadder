import { Bitset, ComplexArray, StringArray } from '../util';

export const enum DataType {
  Int8 = 1,
  Uint8 = 2,
  Uint8Clamped = 4,
  Int16 = 8,
  Uint16 = 16,
  Int32 = 32,
  Uint32 = 64,
  Float32 = 128,
  Float64 = 256,
  Complex = 512,
  Bool = 1024,
  String = 2048,
  Int64 = 4096,
  Uint64 = 8192,
  Any = 16384,
}

export const dataTypeNames = {
  [DataType.Int8]: 'int8',
  [DataType.Uint8]: 'uint8',
  [DataType.Uint8Clamped]: 'uint8-clamped',
  [DataType.Int16]: 'int16',
  [DataType.Uint16]: 'uint16',
  [DataType.Int32]: 'int32',
  [DataType.Uint32]: 'uint32',
  [DataType.Float32]: 'float32',
  [DataType.Float64]: 'float64',
  [DataType.Complex]: 'complex',
  [DataType.Bool]: 'bool',
  [DataType.String]: 'string',
  [DataType.Int64]: 'int64',
  [DataType.Uint64]: 'uint64',
  [DataType.Any]: 'any'
};

export type IntType = 
  | DataType.Int8
  | DataType.Uint8
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
  ? NumericType | DataType.Bool | DataType.String
  : T extends DataType.Any | DataType.String | DataType.Bool
    ? DataType
    : T extends BigNumericType
      ? BigNumericType
      : never;

export function isAssignable<T1 extends DataType>(dst: T1, src: DataType): src is AssignableType<T1> {
  if (dst == src || dst == DataType.Any || dst == DataType.String || dst == DataType.Bool) return true;
  if (dst == DataType.Int64 || dst == DataType.Uint64) return src == DataType.Int64 || src == DataType.Uint64;
  if (dst <= DataType.Float64) return src <= DataType.String;
  return false;
}

export function guessType(value: unknown) {
  if (typeof value == 'number') {
    return Number.isInteger(value) ? DataType.Int32 : DataType.Float64;
  }
  if (typeof value == 'bigint') return DataType.Int64;
  if (typeof value == 'string') return DataType.String;
  if (typeof value == 'boolean') return DataType.Bool;
  if (value && typeof value['real'] == 'number' && typeof value['imag'] == 'number') return DataType.Complex;
  return DataType.Any;
}

// should only be called on return values from guessType for accurate result
export function bestGuess(types: DataType[]) {
  if (types.length < 2) return types[0] || DataType.Float64;
  let maxType = types[0];
  if (maxType >= DataType.Bool) return types.every(v => v == maxType) ? maxType : DataType.Any;
  for (const type of types) {
    if (type > maxType) {
      if (type >= DataType.Bool) return DataType.Any;
      maxType = type;
    }
  }
  return maxType;
}

export const dataTypeBufferMap = {
  [DataType.Int8]: Int8Array,
  [DataType.Uint8]: Uint8Array,
  [DataType.Uint8Clamped]: Uint8ClampedArray,
  [DataType.Int16]: Int16Array,
  [DataType.Uint16]: Uint16Array,
  [DataType.Int32]: Int32Array,
  [DataType.Uint32]: Uint32Array,
  [DataType.Float32]: Float32Array,
  [DataType.Float64]: Float64Array,
  [DataType.Complex]: ComplexArray,
  [DataType.Bool]: Bitset,
  [DataType.String]: StringArray,
  [DataType.Int64]: BigInt64Array,
  [DataType.Uint64]: BigUint64Array,
  [DataType.Any]: Array as { new(length: number): unknown[] }
};

type DTBM = typeof dataTypeBufferMap;

export type DataTypeBuffer<T extends DataType> = InstanceType<DTBM[T]>;
export type InferDataType<T extends DataTypeBuffer<DataType>> = {
  [K in keyof DTBM]: T extends DataTypeBuffer<K> ? K : never
}[keyof DTBM];
export type IndexType<T extends DataType> = DataTypeBuffer<T>[number];
