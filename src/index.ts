import { Bitset } from './util/bitset'

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
  Bool,
  Int64,
  Uint64,
  Object
}

type NumericType = 
  | DataType.Int8
  | DataType.Uint8
  | DataType.Uint8Clamped
  | DataType.Int16
  | DataType.Uint16
  | DataType.Int32
  | DataType.Float32
  | DataType.Float64;

type BigNumericType = DataType.Int64 | DataType.Uint64;

type AssignableType<T extends DataType> = T extends NumericType
  ? NumericType | DataType.Bool
  : T extends DataType.Object
    ? DataType
    : T extends DataType.Bool
      ? DataType.Bool
      : T extends BigNumericType
        ? BigNumericType
        : never;

const isAssignable = <T1 extends DataType>(dst: T1, src: DataType): src is AssignableType<T1> => {
  if (dst == src || dst == DataType.Object) return true;
  if (dst == DataType.Bool) return src == DataType.Bool;
  if (dst == DataType.Int64 || dst == DataType.Uint64) return src == DataType.Int64 || src == DataType.Uint64;
  if (dst <= DataType.Float64) return src <= DataType.Bool;
  return false;
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
  [DataType.Bool]: Bitset,
  [DataType.Object]: Array as { new(length: number): unknown[] }
};

type DataTypeTypeMap = typeof dataTypeBufferMap;

type DataTypeBuffer<T extends DataType> = InstanceType<DataTypeTypeMap[T]>

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

export type Dims = readonly number[];

type IndexType<T extends DataType> = DataTypeBuffer<T>[number];

class FlatArray<T extends DataType> {
  // type
  t: T;
  // buffer
  b: DataTypeBuffer<T>;

  constructor(data: DataTypeBuffer<T>);
  constructor(type: T, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size?: number) {
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](size) as DataTypeBuffer<T>;
      if (this.t == DataType.Object) (this.b as unknown[]).fill(null);
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
    }
  }
}

const rangeSize = (s: number, e: number, t: number) => Math.floor(Math[t > 0 ? 'max' : 'min'](e - s, 0) / t);

const fixInd = (ind: number, size: number, loose?: 1) => {
  if (!Number.isInteger(ind) || (!loose && (ind < -size || ind >= size))) {
    throw new RangeError(`index ${ind} out of bounds in dimension of length ${size}`);
  }
  return ind < 0 ? ind + size : ind;
}

type NDViewChild<T extends DataType, D extends Dims> =
  D extends readonly [number, ...infer NextD]
    ? [] extends NextD
      ? IndexType<T>
      : NDView<T, NextD extends Dims ? NextD : never>
    : IndexType<T> | NDView<T, Dims>;


const numOps = [
  ['add', 'add', (a: number, b: number) => a + b],
  ['subtract', 'sub', (a: number, b: number) => a - b],
  ['multiply', 'mul', (a: number, b: number) => a * b],
  ['divide', 'div', (a: number, b: number) => a / b],
  ['modulo', 'mod', (a: number, b: number) => a % b],
  ['exponentiate', 'pow', (a: number, b: number) => a ** b],
  ['bitwise and', 'bitAnd', (a: number, b: number) => a & b],
  ['bitwise or', 'bitOr', (a: number, b: number) => a | b],
  ['bitwise xor', 'bitXor', (a: number, b: number) => a ^ b],
] as const;

type NumOps = {
  [T in (typeof numOps)[number][1]]: <T extends DataType, D extends Dims>(this: NDView<T, D>, value: NDView<T, D>, inPlace?: boolean) => NDView<T, D>;
}

// zero width space - we'll use it for some fun hacks :P
const zws = String.fromCharCode(0x200B);

const recentAccesses = new Map<number, NDView<DataType, Dims>>();

const getFreeID = () => {
  let id = -1;
  while (recentAccesses.has(++id));
  return id;
}

const indexablePrefix = `ndarray${zws}`

interface NDView<T extends DataType, D extends Dims> extends Iterable<NDViewChild<T, D>>, NumOps {
  [index: number]: NDViewChild<T, D>;
  [index: string]: IndexType<T> | NDView<T, Dims>;
}
class NDView<T extends DataType, D extends Dims> {
  // raw ndarray
  private t: FlatArray<T>;
  // dimensions
  private d: Dims;
  // stride
  private s: number[];
  // offset
  private o: number;

  constructor(src: FlatArray<T>, dims: Dims, stride: number[], offset: number) {
    this.t = src;
    this.d = dims;
    this.s = stride;
    this.o = offset;

    const get = (target: this, key: string | symbol, unbox?: 1) => {
      if (typeof key == 'symbol') return target[key as unknown as string];
      const parts = key.split(',').map(part => part.trim().split(':'));
      let nextSrc = src;
      let nextDims = dims.slice();
      let nextStride = stride.slice();
      let nextOffset = offset;
      let workingIndex = 0;
      for (let i = 0; i < parts.length; ++i) {
        const part = parts[i];
        if (workingIndex >= nextDims.length && (part.length > 1 || part[0] != '+')) {
          throw new TypeError('cannot slice 0-D ndarray');
       }
        if (part.length == 1) {
          if (part[0] == '...') {
            // TODO: improve this block (lots of repeated code)
            workingIndex = nextDims.length;
            for (let j = i + 1; j < parts.length; ++j) {
              const newPart = parts[j];
              if (newPart.length == 1) {
                if (newPart[0] == '...') {
                  throw new TypeError('only one ellipsis allowed in index');
                } else if (newPart[0].startsWith(indexablePrefix)) {
                  let ind = indexablePrefix.length;
                  for (; ind < part[0].length; ++ind) {
                    if (part[0][ind] != zws) break;
                  }
                  const id = ind - indexablePrefix.length;
                  const view = recentAccesses.get(id);
                  if (view) {
                    if (view.t.t <= DataType.Uint32) workingIndex -= 1;
                    else if (view.t.t == DataType.Bool) workingIndex -= view.d.length;
                    else {
                      throw new TypeError(`cannot index ndarray with ndarray of type ${DataType[view.t.t]}`);
                    }
                  } else {
                    throw new TypeError(`cannot index ndarray with ndarray of type ${DataType[view.t.t]}`);
                  }
                }
              } else workingIndex -= 1;
            }
            continue;

          } else if (part[0] == '+') {
            nextDims.splice(workingIndex, 0, 1);
            nextStride.splice(workingIndex, 0, 0);
            workingIndex++;
            continue;
          } else if (part[0].startsWith(indexablePrefix)) {
            let i = indexablePrefix.length;
            for (; i < part[0].length; ++i) {
              if (part[0][i] != zws) break;
            }
            const id = i - indexablePrefix.length;
            const view = recentAccesses.get(id);
            if (view) {
              recentAccesses.delete(id);
              if (view.t.t <= DataType.Uint32) {
                const tmpView = ndarray(nextSrc.t, view.d.concat(nextDims.slice(workingIndex)));

              } else if (view.t.t == DataType.Bool) {
                const preDims = nextDims.slice(0, workingIndex);
                const workingDims = nextDims.slice(workingIndex);
                if (view.d.length != workingDims.length || view.d.some((v, i) => workingDims[i] != v)) {
                  throw new TypeError(`incompatible dimensions: expected (${workingDims.join(', ')}), found (${view.d.join(', ')})`);
                }
                const trueOffsets: number[] = [];
                for (const ind of view.r()) {
                  if (view.t.b[view.c(ind)]) {
                    let offset = nextOffset;
                    for (let i = 0; i < workingDims.length; ++i) offset += ind[i] * nextStride[workingIndex + i];
                    trueOffsets.push(offset);
                  }
                }
                const tmpView = ndarray(nextSrc.t, [...preDims, trueOffsets.length]);
                for (const ind of tmpView.r()) {
                  let offset = trueOffsets[ind[workingIndex]];
                  for (let i = 0; i < workingIndex; ++i) offset += ind[i] * nextStride[i];
                  tmpView.t.b[tmpView.c(ind)] = nextSrc.b[offset];
                }
                nextSrc = tmpView.t;
                // tmpView no longer needed, so can be mutable
                nextDims = tmpView.d as number[];
                nextStride = tmpView.s;
                nextOffset = tmpView.o;
                workingIndex = nextDims.length;
              } else {
                throw new TypeError(`cannot index ndarray with ndarray of type ${DataType[view.t.t]}`);
              }
              continue;
            }
            throw new TypeError('ndarray index expired: ensure slices are used immediately after creation');
          }
          let ind = +part[0];
          if (parts.length == 1 && isNaN(ind)) return target[key];
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
      if (unbox && !nextDims.length) return nextSrc.b[nextOffset];
      return new NDView<T, Dims>(nextSrc, nextDims, nextStride, nextOffset);
    }

    return new Proxy(this, {
      get: (target, key) => get(target, key, 1),
      set: (target, key, value) => {
        if (typeof key == 'symbol' || (!key.includes(':') && isNaN(+key))) {
          target[key as string] = value;
        } else {
          (get(target, key) as NDView<T, Dims>).set(value);
        }
        return true;
      }
    });
  }

  // calculate offset
  private c(ind: number[]) {
    let offset = this.o;
    for (let i = 0; i < ind.length; ++i) offset += ind[i] * this.s[i];
    return offset;
  }

  // symmetric operation preperation
  private y(value: NDView<DataType, D> | IndexType<T>, strict?: boolean | 1) {
    if (!(value instanceof NDView)) {
      const buf = new dataTypeBufferMap[this.t.t](1) as DataTypeBuffer<T>;
      buf[0] = value;
      value = new NDView(new FlatArray(buf), this.d, this.d.map(() => 0), 0);
    }
    const val = value as NDView<T, D>;
    if (strict && !isAssignable(this.t.t, val.t.t)) {
      throw new TypeError(`cannot assign to ndarray of type ${DataType[val.t.t]} to ${DataType[this.t.t]}`);
    }
    if (val.d.length != this.d.length || val.d.some((v, i) => this.d[i] != v)) {
      throw new TypeError(`incompatible dimensions: expected (${this.d.join(', ')}), found (${val.d.join(', ')})`);
    }
    return val;
  }

  // iterate over indices with same array
  private *r() {
    const dims = this.d;
    const coord = dims.map(() => -1);
    const inner = function*(dim: number): Generator<number[]> {
      if (dim == dims.length) yield coord;
      else for (let i = 0; i < dims[dim]; ++i) {
        coord[dim] = i;
        yield* inner(dim + 1);
      }
    }
    yield* inner(0);
  }

  *[Symbol.iterator]() {
    if (this.d.length == 1) {
      for (let i = 0; i < this.d[0]; ++i) {
        yield this.t.b[this.o + i * this.s[0]] as NDViewChild<T, D>;
      }
    } else {
      const nextDims = this.d.slice(1), nextStride = this.s.slice(1);
      for (let i = 0; i < this.d[0]; ++i) {
        yield new NDView<T, Dims>(this.t, nextDims, nextStride, this.o + i * this.s[0]) as NDViewChild<T, D>;
      }
    }
  }

  set(value: NDView<T, D> | IndexType<T>) {
    const val = this.y(value, 1);
    const coord = this.d.map(() => -1);
    const set = (dim: number) => {
      if (dim == this.d.length) {
        this.t.b[this.c(coord)] = val.t.b[val.c(coord)];
      } else if (dim == this.d.length - 1 && this.s[dim] == 1 && val.s[dim] == 1) {
        coord[dim] = 0;
        const srcStart = val.c(coord);
        (this.t.b as Uint8Array).set((val.t.b as Uint8Array).subarray(srcStart, srcStart + this.d[coord.length]), this.c(coord));
      } else {
        for (let i = 0; i < this.d[dim]; ++i) {
          coord[dim] = i;
          set(dim + 1);
        }
      }
    }
    set(0);
  }

  toString() {
    const coord = this.d.map(() => -1);
    const stringify = (dim: number) => {
      if (dim == this.d.length) return this.t.b[this.c(coord)].toString();
      let str = '[';
      for (let i = 0; i < this.d[dim]; ++i) {
        coord[dim] = i;
        str += stringify(dim + 1) + ', ';
      }
      return str.slice(0, -2) + ']';
    }
    return `ndarray<${DataType[this.t.t]}>(${this.d.join(', ')}) ${stringify(0)}`
  }

  static {
    const numOp = (opName: string, name: string, op: (a: number, b: number) => void) => {
      NDView.prototype[name] = function<T extends DataType, D extends Dims>(this: NDView<T, D>, value: NDView<T, D> | IndexType<T>, inPlace?: boolean) {
        if (this.t.t >= DataType.Bool) throw new TypeError(`cannot ${opName} non-numeric ndarrays`);
        const val = this.y(value, inPlace);
        if (inPlace) {
          for (const index of this.r()) {
            const ind = this.c(index);
            this.t.b[ind] = op(this.t.b[ind] as number, val.t.b[val.c(index)] as number);
          }
          return this;
        } else {
          let type: DataType = this.t.t;
          if (!isAssignable(type, val.t.t)) {
            type = val.t.t;
            if (!isAssignable(type, this.t.t)) type = DataType.Float32;
          }
          const dst = ndarray(type, this.d);
          for (const index of this.r()) {
            dst.t.b[dst.c(index)] = op(this.t.b[this.c(index)] as number, val.t.b[val.c(index)] as number);
          }
          return dst;
        }
      };
    }
    for (const op of numOps) numOp.apply(null, op);
  }

  private [Symbol.for('nodejs.util.inspect.custom')]() {
    return this.toString();
  }

  private [Symbol.toPrimitive]() {
    const id = getFreeID();
    recentAccesses.set(id, this);
    queueMicrotask(() => recentAccesses.delete(id));
    return `${indexablePrefix}${zws.repeat(id)}<${DataType[this.t.t]}>(${this.d.join('x')}) [...]`;
  }

  reshape(dims: Dims) {
    if (dims.reduce((a, b) => a * b, 1) != this.size) {
      throw new TypeError(`dimensions (${dims.join(', ')}) do not match data length ${this.size}`);
    }
    const cd: number[] = [], cs: number[] = [];
    for (let i = 0; i < this.d.length; ++i) {
      if (this.d[i] != 1) {
        cd.push(this.d[i]);
        cs.push(this.s[i]);
      }
    }
    
  }

  get shape(): D {
    return this.d.slice() as Dims as D;
  }

  get size(): number {
    return this.d.reduce((a, b) => a * b, 1);
  }
}

export function ndarray<D extends Dims>(data: Int8Array, dimensions: D): NDView<DataType.Int8, D>;
export function ndarray<D extends Dims>(data: Uint8Array, dimensions: D): NDView<DataType.Uint8, D>;
export function ndarray<D extends Dims>(data: Int16Array, dimensions: D): NDView<DataType.Int16, D>;
export function ndarray<D extends Dims>(data: Uint16Array, dimensions: D): NDView<DataType.Uint16, D>;
export function ndarray<D extends Dims>(data: Int32Array, dimensions: D): NDView<DataType.Int32, D>;
export function ndarray<D extends Dims>(data: Uint32Array, dimensions: D): NDView<DataType.Uint32, D>;
export function ndarray<D extends Dims>(data: BigInt64Array, dimensions: D): NDView<DataType.Int64, D>;
export function ndarray<D extends Dims>(data: BigUint64Array, dimensions: D): NDView<DataType.Uint64, D>;
export function ndarray<D extends Dims>(data: Float32Array, dimensions: D): NDView<DataType.Float32, D>;
export function ndarray<D extends Dims>(data: Float64Array, dimensions: D): NDView<DataType.Float64, D>;
export function ndarray<D extends Dims>(data: Bitset, dimensions: D): NDView<DataType.Bool, D>;
export function ndarray<T extends DataType, D extends Dims>(dataOrType: T | DataTypeBuffer<T>, dimensions: D): NDView<T, D>;
export function ndarray<T extends DataType, D extends Dims>(dataOrType: T | DataTypeBuffer<T>, dimensions: D) {
  const size = dimensions.reduce((a, b) => a * b, 1);
  const src = new FlatArray(dataOrType, size);
  if (src.b.length != size) {
    throw new TypeError(`dimensions (${dimensions.join(', ')}) do not match data length ${src.b.length}`);
  }
  const stride: number[] = [];
  let cur = 1;
  for (let i = dimensions.length - 1; i >= 0; --i) {
    stride.unshift(cur);
    cur *= dimensions[i];
  }
  return new NDView<T, D>(src, dimensions, stride, 0);
}

export { Bitset };