export enum DataType {
  Int8,
  Uint8,
  Uint8Clamped,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Int64,
  Uint64,
  Float32,
  Float64,
  Bool,
  String,
  Object
}

const dataTypeBufferMap = {
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
  [DataType.Bool]: Array as { new(length: number): boolean[] },
  [DataType.String]: Array as { new(length: number): string[] },
  [DataType.Object]: Array as { new(length: number): unknown[] }
};

type DataTypeTypeMap = typeof dataTypeBufferMap;

type DataTypeBuffer<T extends DataType> = InstanceType<DataTypeTypeMap[T]>

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  if (Array.isArray(data)) {
    if (data.length == 0) {
      throw new TypeError('cannot infer type from empty array');
    }
    const expect = typeof data[0];
    if (data.some(d => typeof d != expect)) return DataType.Object as T;
    if (expect == 'string') return DataType.String as T;
    if (expect == 'boolean') return DataType.Bool as T;
    return DataType.Object as T;
  }
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

export type Dims = readonly number[];

type IndexType<T extends DataType> = DataTypeBuffer<T>[number];

class RawTensor<T extends DataType> {
  // type
  t: T;
  // buffer
  b: DataTypeBuffer<T>;

  constructor(data: DataTypeBuffer<T>);
  constructor(type: T, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size?: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size?: number) {
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](size) as DataTypeBuffer<T>;
      if (this.t == DataType.String) this.b.fill('' as never);
      else if (this.t == DataType.Bool) this.b.fill(false as never);
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
    }
  }
}

const tvInternals = Symbol('tensorviewinternals');

const rangeSize = (s: number, e: number, t: number) => Math.floor(Math[t > 0 ? 'max' : 'min'](e - s, 0) / t);

const fixInd = (ind: number, size: number, loose?: 1) => {
  if (!Number.isInteger(ind) || (!loose && (ind < -size || ind >= size))) {
    throw new RangeError(`index ${ind} out of bounds in dimension of length ${size}`);
  }
  return ind < 0 ? ind + size : ind;
}

type TensorViewChild<T extends DataType, D extends Dims> =
  D extends readonly [number, ...infer NextD]
    ? [] extends NextD
      ? IndexType<T>
      : TensorView<T, NextD extends Dims ? NextD : never>
    : IndexType<T> | TensorView<T, Dims>;

interface TensorView<T extends DataType, D extends Dims> extends Iterable<TensorViewChild<T, D>> {
  [index: number]: TensorViewChild<T, D>;
  [index: string]: IndexType<T> | TensorView<T, Dims>;
}
class TensorView<T extends DataType, D extends Dims> {
  // proxy bypass
  private [tvInternals]: this;
  // tensor
  private t: RawTensor<T>;
  // dimensions
  private d: Dims;
  // stride
  private s: number[];
  // offset
  private o: number;

  constructor(src: RawTensor<T>, dims: Dims, stride: number[], offset: number) {
    this.t = src;
    this.d = dims;
    this.s = stride;
    this.o = offset;

    return new Proxy(this, {
      get: (target, key) => {
        if (key == tvInternals) return target;
        if (typeof key == 'symbol') return;
        const parts = key.split(',').map(part => part.trim().split(':'));
        let nextDims = dims.slice();
        let nextStride = stride.slice();
        let nextOffset = offset;
        let workingIndex = 0;
        for (let i = 0; i < parts.length; ++i) {
          if (workingIndex >= nextDims.length) {
            throw new TypeError('cannot slice 0-D tensor');
          }
          const part = parts[i];
          if (part.length == 1) {
            if (part[0] == '...') {
              if (parts.slice(i + 1).some(part => part.length == 1 && part[0] == '...')) {
                throw new TypeError('only one ellipsis allowed in index')
              }
              workingIndex = nextDims.length - (parts.length - i - 1);
              continue;
            } else if (part[0] == '+') {
              nextDims.splice(workingIndex, 0, 1);
              nextStride.splice(workingIndex, 0, nextStride[workingIndex++]);
              continue;
            }
            let ind = +part[0];
            ind = fixInd(ind, nextDims.splice(workingIndex, 1)[0]);
            nextOffset += ind * nextStride.splice(workingIndex, 1)[0]
          } else if (part.length > 3) {
            throw new TypeError(`invalid slice ${key}`);
          } else {
            let step = +(part[2] || 1);
            if (step == 0 || !Number.isInteger(step)) {
              throw new TypeError(`invalid step ${step}`);
            }
            const t = step || 1;
            let start = +(part[0] || (step < 0 ? nextDims[workingIndex] - 1 : 0));
            const s = fixInd(start, nextDims[workingIndex], 1);
            const e = part[1]
              ? fixInd(+part[1], nextDims[workingIndex], 1)
              : (step < 0 ? -1 : nextDims[workingIndex]);
            nextOffset += s * nextStride[workingIndex];
            nextDims[workingIndex] = rangeSize(s, e, t);
            nextStride[workingIndex++] *= t;
          }
        }
        if (!nextDims.length) return src.b[nextOffset];
        return new TensorView<T, Dims>(src, nextDims, nextStride, nextOffset);
      },
      set: (target, key, value, receiver) => {
        receiver[key][tvInternals].set(value);
        return true;
      }
    });
  }

  private c(ind: number[]) {
    let offset = this.o;
    for (let i = 0; i < ind.length; ++i) offset += ind[i] * this.s[i];
    return offset;
  }

  set(value: TensorView<T, D>) {
    if (!value[tvInternals]) {
      const buf = new dataTypeBufferMap[this.t.t](1) as DataTypeBuffer<T>;
      buf[0] = value;
      value = new TensorView(new RawTensor(buf), this.d, this.d.map(() => 0), 0);
    }
    const tv = value[tvInternals];
    if (tv.t.t != this.t.t) {
      throw new TypeError('cannot set to tensor of different type');
    }
    if (tv.d.length != this.d.length || tv.d.some((v, i) => this.d[i] != v)) {
      throw new TypeError(`incompatible dimensions: expected (${this.d.join(', ')}), found (${tv.d.join(', ')})`);
    }
    const set = (coord: number[]) => {
      if (coord.length == this.d.length) {
        this.t.b[this.c(coord)] = tv.t.b[tv.c(coord)];
      } else if (coord.length == this.d.length - 1 && this.s[coord.length] == 1 && tv.s[coord.length] == 1) {
        const start = coord.concat(0);
        const srcStart = tv.c(start);
        (this.t.b as Uint8Array).set((tv.t.b as Uint8Array).subarray(srcStart, srcStart + this.d[coord.length]), this.c(start));
      } else {
        for (let i = 0; i < this.d[coord.length]; ++i) set(coord.concat(i));
      }
    }
    set([]);
  }

  toString() {
    const stringify = (coord: number[]) => {
      if (coord.length == this.d.length) return this.t.b[this.c(coord)].toString();
      let str = '[';
      for (let i = 0; i < this.d[coord.length]; ++i) {
        str += stringify(coord.concat(i)) + ', ';
      }
      return str.slice(0, -2) + ']';
    }
    return `Tensor<${DataType[this.t.t]}>(${this.d.join(', ')}) ${stringify([])}`
  }

  *[Symbol.iterator]() {
    if (this.d.length == 1) {
      for (let i = 0; i < this.d[0]; ++i) {
        yield this.t.b[this.o + i * this.s[0]] as TensorViewChild<T, D>;
      }
    } else {
      const nextDims = this.d.slice(1), nextStride = this.s.slice(1);
      for (let i = 0; i < this.d[0]; ++i) {
        yield new TensorView<T, Dims>(this.t, nextDims, nextStride, this.o + i * this.s[0]) as TensorViewChild<T, D>;
      }
    }
  }

  private [Symbol.for('nodejs.util.inspect.custom')]() {
    return (this[tvInternals] || this).toString();
  }

  get shape(): D {
    return this.d.slice() as unknown as D;
  }

  get size(): number {
    return this.d.reduce((a, b) => a * b, 1);
  }
}

export function tensor<D extends Dims>(data: Int8Array, dimensions: D): TensorView<DataType.Int8, D>;
export function tensor<D extends Dims>(data: Uint8Array, dimensions: D): TensorView<DataType.Uint8, D>;
export function tensor<D extends Dims>(data: Int16Array, dimensions: D): TensorView<DataType.Int16, D>;
export function tensor<D extends Dims>(data: Uint16Array, dimensions: D): TensorView<DataType.Uint16, D>;
export function tensor<D extends Dims>(data: Int32Array, dimensions: D): TensorView<DataType.Int32, D>;
export function tensor<D extends Dims>(data: Uint32Array, dimensions: D): TensorView<DataType.Uint32, D>;
export function tensor<D extends Dims>(data: BigInt64Array, dimensions: D): TensorView<DataType.Int64, D>;
export function tensor<D extends Dims>(data: BigUint64Array, dimensions: D): TensorView<DataType.Uint64, D>;
export function tensor<D extends Dims>(data: Float32Array, dimensions: D): TensorView<DataType.Float32, D>;
export function tensor<D extends Dims>(data: Float64Array, dimensions: D): TensorView<DataType.Float64, D>;
export function tensor<D extends Dims>(data: boolean[], dimensions: D): TensorView<DataType.Bool, D>;
export function tensor<D extends Dims>(data: string[], dimensions: D): TensorView<DataType.String, D>;
export function tensor<T extends DataType, D extends Dims>(dataOrType: T | DataTypeBuffer<T>, dimensions: D): TensorView<T, D>;
export function tensor<T extends DataType, D extends Dims>(dataOrType: T | DataTypeBuffer<T>, dimensions: D) {
  const src = new RawTensor(dataOrType, dimensions.reduce((a, b) => a * b, 1));
  const stride: number[] = [];
  let cur = 1;
  for (let i = dimensions.length - 1; i >= 0; --i) {
    stride.unshift(cur);
    cur *= dimensions[i];
  }
  return new TensorView<T, D>(src, dimensions, stride, 0);
}