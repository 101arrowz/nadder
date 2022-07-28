export enum DataType {
  Int8,
  Uint8,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Int64,
  Uint64,
  Float16,
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
  [DataType.Float16]: Float32Array,
  [DataType.Float32]: Float32Array,
  [DataType.Float64]: Float64Array,
  [DataType.Bool]: Array as { new(length: number): boolean[] },
  [DataType.String]: Array as { new(length: number): string[] },
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

type IndexType<T extends DataType> = DataTypeBuffer<T>[number];

class Tensor<T extends DataType, D extends number[]> {
  // type
  t: T;
  // buffer
  b: DataTypeBuffer<T>;
  // dimensions
  d: D;
  // size
  s: number;

  constructor(data: DataTypeBuffer<T>, dimensions: D);
  constructor(type: T, dimensions: D);
  constructor(dataOrType: T | DataTypeBuffer<T>, dimensions: D);
  constructor(dataOrType: T | DataTypeBuffer<T>, dimensions: D) {
    this.s = dimensions.reduce((a, b) => a * b, 1);
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](this.s) as DataTypeBuffer<T>;
      if (this.t == DataType.String) this.b.fill('' as never);
      else if (this.t == DataType.Bool) this.b.fill(false as never);
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
      if (this.b.length != this.s) {
        throw new TypeError(`expected buffer of length ${this.s}, found ${this.b.length}`);
      }
    }
    this.d = dimensions;
  }
}

const tvInternals = Symbol('tensorview-internals');

class TensorView<T extends DataType, D extends number[]> {
  // proxy bypass
  [tvInternals]: this;
  // tensor/view
  private t: Tensor<T, number[]> | TensorView<T, number[]>;
  // index
  private i: number;
  // dimensions
  private d: number[];

  constructor(src: TensorView<T, number[]> | Tensor<T, number[]>, index = -1) {
    this.i = index;
    this.t = src;
    this.d = (src as TensorView<T, number[]>).d.slice(index == -1 ? 0 : 1);
    
    const validate = (ind: number) => {
      if (ind < 0 || ind >= this.d[0] || !Number.isInteger(ind)) {
        throw new TypeError(`invalid index ${ind} into dimension ${this.d[0]}`);
      }
    }
    return new Proxy(this, {
      get: (target, key) => {
        if (key == tvInternals) return this;
        const ind = +(key as string);
        validate(ind);
        if (this.d.length == 1) {
          const path = this.c();
          return path.t.b[path.b + ind];
        }
        return new TensorView<T, D>(target, ind);
      },
      set: (target, key, value) => {
        const ind = +(key as string);
        validate(ind);
        if (!value[tvInternals]) throw new TypeError('value must be a tensor');
        const tv = (value as TensorView<T, number[]>)[tvInternals];
        if (tv.d.length != this.d.length - 1 || tv.d.some((d, i) => d != this.d[i + 1])) {
          throw new TypeError(`invalid tensor dimensions (${tv.d.join(', ')}); expected (${this.d.slice(1).join(', ')})`);
        }
        const path = this.c();
        const src = tv.c();
        if (!tv.d.length) {
          path.t.b[path.b + ind] = src.t.b[src.b];
          return true;
        }
        const pi = path.b + ind;
        const setDims = (offSrc: number, offDst: number, dims: number[]) => {
          const [dim, ...newDims] = dims;
          if (!newDims.length) {
            path.t.b.set(src.t.b.subarray(offSrc * dim, (offSrc + 1) * dim), offDst * dim);
          } else {
            const baseSrc = offSrc * dim, baseDst = offDst * dim;
            for (let i = 0; i < dim; ++i) {
              setDims(baseSrc + i, baseDst + i, newDims);
            }
          }
        }
        setDims(src.b / tv.d[0], pi, tv.d);
        return true;
      }
    })
  }

  // calculate index
  private c() {
    if (typeof (this.t as Tensor<T, D>).t == 'number') {
      return {
        b: 0,
        t: this.t as Tensor<T, number[]>
      };
    }
    const path = (this.t as TensorView<T, number[]>).c();
    return {
      b: (path.b + this.i) * this.d[0],
      t: path.t
    };
  }

  get shape(): D {
    return this.d.slice() as D;
  }
}


export function tensor<T extends DataType, D extends number[]>(dataOrType: T | DataTypeBuffer<T>, dimensions: D) {
  const src = new Tensor(dataOrType, dimensions);
  return new TensorView(src);
}