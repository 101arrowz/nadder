export enum DataType {
  Int8,
  Uint8,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Int64,
  Uint64,
  Float32,
  Float64,
  Bool,
  String
}

const dataTypeBufferMap = {
  [DataType.Int8]: Int8Array,
  [DataType.Uint8]: Uint8Array,
  [DataType.Int16]: Int16Array,
  [DataType.Uint16]: Uint16Array,
  [DataType.Int32]: Int32Array,
  [DataType.Uint32]: Uint32Array,
  [DataType.Int64]: BigInt64Array,
  [DataType.Uint64]: BigUint64Array,
  [DataType.Float32]: Float32Array,
  [DataType.Float64]: Float64Array,
  [DataType.Bool]: Array as { new(length: number): boolean[] },
  [DataType.String]: Array as { new(length: number): string[] }
};

type DataTypeTypeMap = typeof dataTypeBufferMap;

type DataTypeBuffer<T extends DataType> = InstanceType<DataTypeTypeMap[T]>

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  if (Array.isArray(data)) {
    if (data.length == 0) {
      throw new TypeError('cannot infer type from empty array');
    }
    const expect = typeof data[0];
    if (data.some(d => typeof d != expect)) {
      throw new TypeError('array contains mixed types');
    }
    if (expect == 'string') return DataType.String as T;
    if (expect == 'boolean') return DataType.Bool as T;
  }
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

export type Dims = readonly number[];

type IndexType<T extends DataType> = DataTypeBuffer<T>[number];

class RawTensor<T extends DataType, D extends Dims> {
  // type
  t: T;
  // buffer
  b: DataTypeBuffer<T>;
  // dimensions
  d: D;

  constructor(data: DataTypeBuffer<T>, dimensions: D);
  constructor(type: T, dimensions: D);
  constructor(dataOrType: T | DataTypeBuffer<T>, dimensions: D);
  constructor(dataOrType: T | DataTypeBuffer<T>, dimensions: D) {
    const size = dimensions.reduce((a, b) => a * b, 1);
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](size) as DataTypeBuffer<T>;
      if (this.t == DataType.String) this.b.fill('' as never);
      else if (this.t == DataType.Bool) this.b.fill(false as never);
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
      if (this.b.length != size) {
        throw new TypeError(`expected buffer of length ${size}, found ${this.b.length}`);
      }
    }
    this.d = dimensions;
  }
}

const tvInternals = Symbol('tensorview-internals');
const inspect = Symbol.for('nodejs.util.inspect.custom');

interface TensorView<T extends DataType, D extends Dims> {
  [index: number]: D extends readonly [number, ...infer NextD]
    ? [] extends NextD
      ? IndexType<T>
      : TensorView<T, NextD extends Dims ? NextD : never>
    :  IndexType<T> | TensorView<T, readonly number[]>;
}
class TensorView<T extends DataType, D extends Dims> {
  // proxy bypass
  private [tvInternals]: this;
  // tensor/view
  private t: RawTensor<T, Dims> | TensorView<T, Dims>;
  // index
  private i: number;
  // dimensions
  private d: Dims;

  constructor(src: TensorView<T, Dims> | RawTensor<T, Dims>, index = -1) {
    this.i = index;
    this.t = src;
    this.d = (src as TensorView<T, Dims>).d.slice(index == -1 ? 0 : 1);
    
    return new Proxy(this, {
      get: (target, key) => {
        if (key == tvInternals) return target;
        if (typeof key == 'symbol') return;
        let ind = +key;
        ind = target.v(ind);
        return target.g(ind);
      },
      set: (target, key, value) => {
        const ind = target.v(+(key as string));
        target.s(ind, value);
        return true;
      }
    });
  }

  // validate index
  private v(ind: number) {
    if (ind < -this.d[0] || ind >= this.d[0] || !Number.isInteger(ind)) {
      throw new TypeError(`invalid index ${ind} into dimension of size ${this.d[0]}`);
    }
    if (ind < 0) ind += this.d[0];
    return ind;
  }

  // get 
  private g(ind: number): IndexType<T> | TensorView<T, Dims> {
    if (this.d.length == 1) {
      const path = this.c();
      return path.t.b[path.b + ind];
    }
    return new TensorView<T, Dims>(this, ind);
  }

  // set
  private s(ind: number, value: unknown) {
    const path = this.c();
    if (!value[tvInternals]) {
      const raw = new RawTensor<T, Dims>(path.t.t, []);
      raw.b[0] = value as never;
      value = new TensorView(raw);
    }
    const tv = (value as TensorView<T, Dims>)[tvInternals];
    if (tv.d.length != this.d.length - 1 || tv.d.some((d, i) => d != this.d[i + 1])) {
      throw new TypeError(`invalid tensor dimensions (${tv.d.join(', ')}); expected (${this.d.slice(1).join(', ')})`);
    }
    const src = tv.c();
    if (path.t.t != src.t.t) {
      throw new TypeError(`incompatible types: ${DataType[path.t.t]} != ${DataType[src.t.t]}`);
    }
    if (!tv.d.length) {
      path.t.b[path.b + ind] = src.t.b[src.b];
      return;
    }
    const pi = path.b + ind;
    const setDims = (offSrc: number, offDst: number, dims: Dims) => {
      const [dim, ...newDims] = dims;
      const baseSrc = offSrc * dim, baseDst = offDst * dim;
      if (!newDims.length) {
        if ((path.t.b as Uint8Array).set) {
          (path.t.b as Uint8Array).set((src.t.b as Uint8Array).subarray(baseSrc, baseSrc + dim), baseDst);
        } else {
          for (let i = 0; i < dim; ++i) {
            path.t.b[baseDst + i] = src.t.b[baseSrc + i];
          }
        }
      } else {
        for (let i = 0; i < dim; ++i) {
          setDims(baseSrc + i, baseDst + i, newDims);
        }
      }
    }
    setDims(src.b / tv.d[0], pi, tv.d);
  }

  // calculate index
  private c(): { b: number; t: RawTensor<T, Dims> } {
    if (typeof (this.t as RawTensor<T, D>).t == 'number') {
      return {
        b: 0,
        t: this.t as RawTensor<T, Dims>
      };
    }
    const path = (this.t as TensorView<T, Dims>).c();
    return {
      b: (path.b + this.i) * this.d[0],
      t: path.t
    };
  }

  private r() {
    let str = '[';
    for (let i = 0; i < this.d[0]; ++i) {
      const val = this.g(i);
      str += (this.d.length > 1 ? (val as TensorView<T, Dims>)[tvInternals].r() : JSON.stringify(val)) + ', ';
    }
    return str.slice(0, -2) + ']';
  }

  toString() {
    if (!this.d.length) return this.g(0).toString();
    let str = `Tensor<${DataType[this.c().t.t]}>(${this.d.join(', ')}) [`;
    for (let i = 0; i < this.d[0]; ++i) {
      const val = this.g(i);
      str += `${(this.d.length > 1 ? (val as TensorView<T, Dims>)[tvInternals].r() : JSON.stringify(val))}, `;
    }
    return str.slice(0, -2) + ']';
  }

  [inspect]() {
    return this[tvInternals].toString();
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
  const src = new RawTensor(dataOrType, dimensions);
  return new TensorView<T, D>(src);
}