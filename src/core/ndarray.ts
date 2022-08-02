import { DataType, DataTypeBuffer, dataTypeNames, dataTypeBufferMap, IndexType, isAssignable, bestGuess, guessType } from './datatype';
import { FlatArray } from './flatarray';
import { Bitset, Complex, ComplexArray, StringArray } from '../util';

export type Dims = readonly number[];

const fixInd = (ind: number, size: number, loose?: 1) => {
  if (!Number.isInteger(ind) || (!loose && (ind < -size || ind >= size))) {
    throw new RangeError(`index ${ind} out of bounds in dimension of length ${size}`);
  }
  return ind < 0 ? ind + size : ind;
}

type NDViewChild<T extends DataType, D extends Dims> =
  D extends readonly []
    ? never
    : D extends readonly [number, ...infer NextD]
      ? [] extends NextD
        ? IndexType<T>
        : NDView<T, NextD extends Dims ? NextD : never>
      : IndexType<T> | NDView<T, Dims>;


// zero width space - we'll use it for some fun hacks :P
const zws = String.fromCharCode(0x200B);

const recentAccesses = new Map<number, NDView<DataType, Dims>>();

const getFreeID = () => {
  let id = -1;
  while (recentAccesses.has(++id));
  return id;
}

const indexablePrefix = `ndarray${zws}`

export interface NDView<T extends DataType = DataType, D extends Dims = Dims> extends Iterable<NDViewChild<T, D>> {
  [index: number]: NDViewChild<T, D>;
  [index: string]: IndexType<T> | NDView<T, Dims>;
}
export class NDView<T extends DataType, D extends Dims> {
  // raw ndarray
  private t: FlatArray<T>;
  // dimensions
  private d: D;
  // stride
  private s: number[];
  // offset
  private o: number;

  constructor(src: FlatArray<T>, dims: D, stride: number[], offset: number) {
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
                      throw new TypeError(`cannot index ndarray with ndarray of type ${dataTypeNames[view.t.t]}`);
                    }
                  } else {
                    throw new TypeError(`cannot index ndarray with ndarray of type ${dataTypeNames[view.t.t]}`);
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
            if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0-D ndarray');
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
                nextDims = tmpView.d;
                nextStride = tmpView.s;
                nextOffset = tmpView.o;
                workingIndex = nextDims.length;
              } else {
                throw new TypeError(`cannot index ndarray with ndarray of type ${dataTypeNames[view.t.t]}`);
              }
              continue;
            }
            throw new TypeError('ndarray index expired: ensure slices are used immediately after creation');
          }
          let ind = +part[0];
          if (parts.length == 1 && isNaN(ind)) return target[key];
          if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0-D ndarray');
          ind = fixInd(ind, nextDims.splice(workingIndex, 1)[0]);
          nextOffset += ind * nextStride.splice(workingIndex, 1)[0]
        } else if (part.length > 3) {
          throw new TypeError(`invalid slice ${key}`);
        } else {
          if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0-D ndarray');
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
          nextDims[workingIndex] = Math.floor(Math[t > 0 ? 'max' : 'min'](e - s, 0) / t);
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
      throw new TypeError(`cannot assign to ndarray of type ${dataTypeNames[val.t.t]} to ${dataTypeNames[this.t.t]}`);
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
    if (!this.d.length) throw new TypeError('cannot iterate over scalar');
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
    return `ndarray<${dataTypeNames[this.t.t]}>(${this.d.join(', ')}) ${stringify(0)}`
  }

  // static {
  //   const numOp = (opName: string, name: string, op: (a: number, b: number) => void) => {
  //     NDView.prototype[name] = function<T extends DataType, D extends Dims>(this: NDView<T, D>, value: NDView<T, D> | IndexType<T>, inPlace?: boolean) {
  //       if (this.t.t >= DataType.Bool) throw new TypeError(`cannot ${opName} non-numeric ndarrays`);
  //       const val = this.y(value, inPlace);
  //       if (inPlace) {
  //         for (const index of this.r()) {
  //           const ind = this.c(index);
  //           this.t.b[ind] = op(this.t.b[ind] as number, val.t.b[val.c(index)] as number);
  //         }
  //         return this;
  //       } else {
  //         let type: DataType = this.t.t;
  //         if (!isAssignable(type, val.t.t)) {
  //           type = val.t.t;
  //           if (!isAssignable(type, this.t.t)) type = DataType.Float32;
  //         }
  //         const dst = ndarray(type, this.d);
  //         for (const index of this.r()) {
  //           dst.t.b[dst.c(index)] = op(this.t.b[this.c(index)] as number, val.t.b[val.c(index)] as number);
  //         }
  //         return dst;
  //       }
  //     };
  //   }
  //   for (const op of numOps) numOp.apply(null, op);
  // }

  private [Symbol.for('nodejs.util.inspect.custom')]() {
    return this.toString();
  }

  private [Symbol.toPrimitive]() {
    const id = getFreeID();
    recentAccesses.set(id, this);
    queueMicrotask(() => recentAccesses.delete(id));
    return `${indexablePrefix}${zws.repeat(id)}<${dataTypeNames[this.t.t]}>(${this.d.join('x')}) [...]`;
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

  get dtype(): DataType {
    return this.t.t;
  }
}

export function ndarray<D extends Dims>(data: Int8Array, dimensions: D): NDView<DataType.Int8, D>;
export function ndarray<D extends Dims>(data: Uint8Array, dimensions: D): NDView<DataType.Uint8, D>;
export function ndarray<D extends Dims>(data: Uint8ClampedArray, dimensions: D): NDView<DataType.Uint8Clamped, D>;
export function ndarray<D extends Dims>(data: Int16Array, dimensions: D): NDView<DataType.Int16, D>;
export function ndarray<D extends Dims>(data: Uint16Array, dimensions: D): NDView<DataType.Uint16, D>;
export function ndarray<D extends Dims>(data: Int32Array, dimensions: D): NDView<DataType.Int32, D>;
export function ndarray<D extends Dims>(data: Uint32Array, dimensions: D): NDView<DataType.Uint32, D>;
export function ndarray<D extends Dims>(data: Float32Array, dimensions: D): NDView<DataType.Float32, D>;
export function ndarray<D extends Dims>(data: Float64Array, dimensions: D): NDView<DataType.Float64, D>;
export function ndarray<D extends Dims>(data: ComplexArray, dimensions: D): NDView<DataType.Complex, D>;
export function ndarray<D extends Dims>(data: Bitset, dimensions: D): NDView<DataType.Bool, D>;
export function ndarray<D extends Dims>(data: StringArray, dimensions: D): NDView<DataType.String, D>;
export function ndarray<D extends Dims>(data: BigInt64Array, dimensions: D): NDView<DataType.Int64, D>;
export function ndarray<D extends Dims>(data: BigUint64Array, dimensions: D): NDView<DataType.Uint64, D>;
export function ndarray<D extends Dims>(data: unknown[], dimensions: D): NDView<DataType.Any, D>;
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

type RecursiveArray<T> = T | RecursiveArray<T>[];


const recurseFind = (data: RecursiveArray<unknown>): [number[], unknown[], DataType] => {
  if (Array.isArray(data)) {
    if (data.length == 0) return [[0], [], DataType.Any];
    const results = data.map(recurseFind);
    const newType = bestGuess(results.map(([,,t]) => t));
    if (results.some(([dim]) => dim.length != results[0][0].length || dim.some((v, i) => results[0][0][i] != v))) {
      throw new TypeError('jagged ndarrays are not supported');
    }
    return [[data.length, ...results[0][0]], results.flatMap(([,b]) => b), newType];
  }
  return [[], [data], guessType(data)];
}

export function array(data: RecursiveArray<number>): NDView<DataType.Int32 | DataType.Float64>;
export function array(data: RecursiveArray<bigint>): NDView<DataType.Int64>;
export function array(data: RecursiveArray<string>): NDView<DataType.String>;
export function array(data: RecursiveArray<boolean>): NDView<DataType.Bool>;
export function array(data: RecursiveArray<Complex>): NDView<DataType.Complex>;
export function array(data: RecursiveArray<unknown>): NDView<DataType.Any>;
export function array(data: RecursiveArray<unknown>): NDView<DataType> {
  const [dims, flat, type] = recurseFind(data);
  const arr = ndarray(type, dims);
  for (let i = 0; i < flat.length; ++i) {
    arr['t'].b[i] = flat[i];
  }
  return arr;
}
