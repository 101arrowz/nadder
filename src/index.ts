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
  private t: T;
  // buffer
  private b: DataTypeBuffer<T>;
  // dimensions
  private d: D;
  private s: number;

  constructor(data: DataTypeBuffer<T>, dimensions: D);
  constructor(type: T, dimensions: D);
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
      this.d = dimensions;
    }
  }
}

class TensorView<T extends DataType, D extends number[]> {
  // tensor/view
  private t: Tensor<T, number[]> | TensorView<T, number[]>;
  // index
  private i: number;
  // dimensions
  private d: number[];

  private constructor(src: TensorView<T, number[]>, index: number, set?: TensorView<T, number[]>) {
    this.t = src;
    this.i = index;
    this.d = src.d.slice(1);
    if (set) {
      // const path = this.c();
      // const rankSet = (rank: number[]) => {
      //   rankSet()
      // }
      throw new Error('help')
      return true as unknown as TensorView<T, D>;
    }
    if (!this.d.length) {
      const path = this.c();
      return path.t[path.b];
    }
    const validate = (ind: number) => {
      if (ind < 0 || ind >= this.d[0] || !Number.isInteger(ind)) {
        throw new TypeError(`invalid index ${ind} into dimension ${this.d[0]}`);
      }
    }
    return new Proxy(this, {
      get: (target, key) => {
        const ind = +(key as string);
        validate(ind);
        return new TensorView<T, D>(target, ind);
      },
      set: (target, key, value) => {
        const ind = +(key as string);
        validate(ind);
        if (!(value instanceof TensorView)) {
        }
        return (new TensorView<T, D>(target, ind, value)) as unknown as boolean;
      }
    })
  }

  // calculate index
  private c() {
    return typeof (this.t as Tensor<T, D>)['t'] == 'number' ? {
      b: 0,
      t: this.t as Tensor<T, number[]>
    } : {
      b: (this.t as TensorView<T, number[]>).c() + this.i * (this.d.length ? this.d[0] : 1),
      t: (this.t as TensorView<T, number[]>).t as Tensor<T, number[]>
    };
  }

  get dims(): D {
    return this.d.slice() as D;
  }
}