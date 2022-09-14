import { DataType, DataTypeBuffer, dataTypeBufferMap } from './datatype';
import { free, malloc, unwatch, wasmExports, watchResize } from '../wasm';
import { Bitset } from '../util';

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

const aligns: Partial<Record<DataType, number>> = {
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
  constructor(dataOrType: T | DataTypeBuffer<T>, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size?: number) {
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](size) as DataTypeBuffer<T>;
      if (this.t == DataType.Any) (this.b as unknown[]).fill(undefined);
      this.l = 0;
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
      this.l = size || 0;
      if (aligns[this.t] && this.l) {
        let reconstruct: () => void;
        if (this.t == DataType.Bool) {
          const { buffer, byteOffset, length } = (this.b as Bitset).buffer;
          if (buffer == wasmExports.memory.buffer) {
            reconstruct = () => {
              let buf = new Uint8Array(wasmExports.memory.buffer, byteOffset, length);
              this.b = new Bitset(buf, (this.b as Bitset).length, (this.b as Bitset).offset) as DataTypeBuffer<T>;
            }
          }
        } else {
          const { buffer, byteOffset, length } = this.b as Uint8Array;
          if (buffer == wasmExports.memory.buffer) {
            reconstruct = () => {
              const buf = new (dataTypeBufferMap[this.t] as Uint8ArrayConstructor)(wasmExports.memory.buffer, byteOffset, length);
              this.b = buf as DataTypeBuffer<T>;
            }
          }
        }
        if (reconstruct) watchResize(this.l, reconstruct);
      }
    }
  }

  w() {
    if (this.l || !wasmExports || !aligns[this.t]) return this.l;
    let allocLength = this.t == DataType.Bool ? (this.b as Bitset).buffer.length : (this.b as Uint8Array).byteLength;
    this.l = malloc(allocLength, aligns[this.t]);
    let reconstruct: () => void;
    if (this.l) {
      if (this.t == DataType.Bool) {
        new Uint8Array(wasmExports.memory.buffer, this.l, allocLength).set((this.b as Bitset).buffer);
        reconstruct = () => {
          let buf = new Uint8Array(wasmExports.memory.buffer, this.l, allocLength);
          this.b = new Bitset(buf, (this.b as Bitset).length, (this.b as Bitset).offset) as DataTypeBuffer<T>;
        }
      } else {
        new (dataTypeBufferMap[this.t] as Uint8ArrayConstructor)(wasmExports.memory.buffer, this.l, this.b.length).set(this.b as Uint8Array);
        reconstruct = () => {
          const buf = new (dataTypeBufferMap[this.t] as Uint8ArrayConstructor)(wasmExports.memory.buffer, this.l, this.b.length);
          this.b = buf as DataTypeBuffer<T>;
        }
      }
      reconstruct();
      watchResize(this.l, reconstruct);
    }
    return this.l;
  }

  f() {
    if (this.l) {
      free(this.l, this.b.length, aligns[this.t]);
      unwatch(this.l);
      this.l = 0;
    }
  }
}