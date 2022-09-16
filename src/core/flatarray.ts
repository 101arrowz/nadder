import { DataType, DataTypeBuffer, dataTypeBufferMap } from './datatype';
import { free, malloc, releaseAlloc, tagAlloc, wasmExports } from '../wasm';
import { Bitset, globalOptions } from '../util';

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

export const aligns: Partial<Record<DataType, number>> = {
  [DataType.Int8]: 1,
  [DataType.Uint8]: 1,
  [DataType.Uint8Clamped]: 1,
  [DataType.Int16]: 2,
  [DataType.Uint16]: 2,
  [DataType.Int32]: 4,
  [DataType.Uint32]: 4,
  [DataType.Float32]: 4,
  [DataType.Float64]: 8,
  [DataType.Bool]: 4,
  [DataType.Int64]: 8,
  [DataType.Uint64]: 8,
};

/** @internal */
export class FlatArray<T extends DataType> {
  // type
  t: T;
  // buffer
  b: DataTypeBuffer<T>;
  // wasm location
  l: number;

  constructor(data: DataTypeBuffer<T>, wasmLoc?: number);
  constructor(type: T, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, sizeOrLoc?: number) {
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      if (globalOptions.preferWASM && aligns[dataOrType] && wasmExports) {
        const allocLength = this.t == DataType.Bool ? (sizeOrLoc + 7) >> 3 : sizeOrLoc * (dataTypeBufferMap[this.t] as Uint8ArrayConstructor).BYTES_PER_ELEMENT;
        this.l = malloc(allocLength, aligns[this.t]);
        if (this.t == DataType.Bool) {
          const nb = new Uint8Array(wasmExports.memory.buffer, this.l, allocLength);
          this.b = new Bitset(nb, sizeOrLoc, 0) as DataTypeBuffer<T>;
        } else {
          this.b = new (dataTypeBufferMap[this.t] as Uint8ArrayConstructor)(wasmExports.memory.buffer, this.l, sizeOrLoc) as DataTypeBuffer<T>;
        }
        tagAlloc(this, allocLength);
      } else {
        this.b = new dataTypeBufferMap[this.t](sizeOrLoc) as DataTypeBuffer<T>;
        if (this.t == DataType.Any) (this.b as unknown[]).fill(undefined);
        this.l = 0;
      }
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
      this.l = sizeOrLoc || 0;
      if (this.l) {
        tagAlloc(this, this.t == DataType.Bool ? (this.b as Bitset).buffer.length : (this.b as Uint8Array).byteLength);
      }
    }
  }

  w() {
    if (this.l || !wasmExports || !aligns[this.t]) return this.l;
    const allocLength = this.t == DataType.Bool ? (this.b as Bitset).buffer.length : (this.b as Uint8Array).byteLength;
    this.l = malloc(allocLength, aligns[this.t]);
    if (this.l) {
      if (this.t == DataType.Bool) {
        const nb = new Uint8Array(wasmExports.memory.buffer, this.l, allocLength);
        nb.set((this.b as Bitset).buffer);
        this.b = new Bitset(nb, this.b.length, (this.b as Bitset).offset) as DataTypeBuffer<T>;
      } else {
        const nb = new (dataTypeBufferMap[this.t] as Uint8ArrayConstructor)(wasmExports.memory.buffer, this.l, this.b.length);
        nb.set(this.b as Uint8Array);
        this.b = nb as DataTypeBuffer<T>;
      }
      tagAlloc(this, allocLength);
    }
    return this.l;
  }

  f() {
    if (this.l) {
      if (this.t == DataType.Bool) {
        this.b = new Bitset((this.b as Bitset).buffer.slice(), this.b.length, (this.b as Bitset).offset) as DataTypeBuffer<T>;
      } else {
        this.b = (this.b as Uint8Array).slice() as DataTypeBuffer<T>;
      }
      releaseAlloc(this.l);
      this.l = 0;
    }
  }
}