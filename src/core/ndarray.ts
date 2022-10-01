import {
  DataType, DataTypeBuffer, dataTypeNames, IndexType, isAssignable, AssignableType
} from './datatype';
import { FlatArray } from './flatarray';
import {
  Bitset, ComplexArray, ndvInternals, broadcast, Broadcastable, ndarray, RecursiveArray, fixInd
} from '../util';

export type Dims = readonly number[];

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

const indexablePrefix = `ndarray${zws}`;

/**
 * An N-dimensional view of a contiguous block of memory, i.e. an ndarray
 */
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

  /** @internal */
  constructor(src: FlatArray<T>, dims: D, stride: number[], offset: number) {
    this.t = src;
    this.d = dims;
    this.s = stride;
    this.o = offset;

    const get = (target: this, key: string | symbol, unbox?: 1) => {
      if (key == ndvInternals) return target;
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
                  throw new SyntaxError('only one ellipsis allowed in index');
                } else if (newPart[0].startsWith(indexablePrefix)) {
                  let ind = indexablePrefix.length;
                  for (; ind < part[0].length; ++ind) {
                    if (part[0][ind] != zws) break;
                  }
                  const id = ind - indexablePrefix.length;
                  const view = recentAccesses.get(id);
                  if (view) {
                    if (view.t.t <= DataType.Uint32) workingIndex -= 1;
                    else if (view.t.t == DataType.Bool) workingIndex -= view.ndim;
                    else {
                      throw new TypeError(`cannot index ndarray with ndarray of type ${dataTypeNames[view.t.t]}`);
                    }
                  } else {
                    throw new TypeError(`cannot index ndarray with ndarray of type ${dataTypeNames[view.t.t]}`);
                  }
                } else if (newPart[0] != '+' && newPart[0] != 'true' && newPart[0] != 'false') {
                  workingIndex -= 1;
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
                if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0D ndarray');
                if (!view.ndim) {
                  let ind = view.t.b[view.o] as number;
                  ind = fixInd(ind, nextDims.splice(workingIndex, 1)[0]);
                  nextOffset += ind * nextStride.splice(workingIndex, 1)[0];
                  continue;
                }
                const preDims = nextDims.slice(0, workingIndex);
                const postDims = nextDims.slice(workingIndex + 1);
                const inView = workingIndex + view.ndim - 1;
                const tmpView = ndarray(nextSrc.t, [...preDims, ...view.d, ...postDims]);
                const copy = (dim: number, viewOff: number, dst: number, src: number) => {
                  if (dim == tmpView.d.length) {
                    tmpView.t.b[dst] = nextSrc.b[src];
                  } else if (dim > inView) {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      copy(dim + 1, viewOff, dst, src);
                      dst += tmpView.s[dim];
                      src += nextStride[dim - view.d.length + 1];
                    }
                  } else if (dim == inView) {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      let ind = view.t.b[viewOff] as number;
                      ind = fixInd(ind, nextDims[workingIndex]);
                      copy(dim + 1, viewOff, dst, src + ind * nextStride[workingIndex]);
                      dst += tmpView.s[dim];
                      viewOff += view.s[dim - workingIndex];
                    }
                  } else if (dim >= workingIndex) {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      copy(dim + 1, viewOff, dst, src);
                      dst += tmpView.s[dim];
                      viewOff += view.s[dim - workingIndex];
                    }
                  } else {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      copy(dim + 1, viewOff, dst, src);
                      dst += tmpView.s[dim];
                      src += nextStride[dim];
                    }
                  }
                }
                copy(0, view.o, 0, nextOffset);
                nextSrc = tmpView.t;
                nextDims = tmpView.d;
                nextStride = tmpView.s;
                nextOffset = tmpView.o;
                workingIndex += view.ndim;
              } else if (view.t.t == DataType.Bool) {
                if (!view.ndim) {
                  nextDims.splice(workingIndex, 0, +view.t.b[view.o]);
                  nextStride.splice(workingIndex, 0, 0);
                  workingIndex++;
                  continue;
                }
                if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0D ndarray');
                const preDims = nextDims.slice(0, workingIndex);
                const workingDims = nextDims.slice(workingIndex);
                if (view.ndim > workingDims.length || view.d.some((v, i) => workingDims[i] != v)) {
                  throw new TypeError(`incompatible dimensions: expected (${workingDims.slice(0, view.ndim).join(', ')}), found (${view.d.join(', ')})`);
                }
                const postDims = workingDims.slice(view.ndim);
                const trueOffsets: number[] = [];
                const collect = (dim: number, ind: number, viewInd: number) => {
                  if (dim == view.d.length) {
                    if (view.t.b[viewInd]) trueOffsets.push(ind);
                  } else {
                    for (let i = 0; i < view.d[dim]; ++i) {
                      collect(dim + 1, ind, viewInd);
                      ind += nextStride[workingIndex + dim];
                      viewInd += view.s[dim];
                    }
                  }
                }
                collect(0, nextOffset, view.o);
                // fix dims todo
                const tmpView = ndarray(nextSrc.t, [...preDims, trueOffsets.length, ...postDims])[ndvInternals];
                const copy = (dim: number, base: number, tmpInd: number) => {
                  if (dim == tmpView.d.length) {
                    tmpView.t.b[tmpInd] = nextSrc.b[base];
                  } else if (dim < workingIndex || dim >= workingIndex + view.d.length) {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      copy(dim + 1, base, tmpInd);
                      base += nextStride[dim];
                      tmpInd += tmpView.s[dim];
                    }
                  } else if (dim == workingIndex) {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      copy(dim + 1, base + trueOffsets[i], tmpInd);
                      tmpInd += tmpView.s[dim];
                    }
                  } else {
                    for (let i = 0; i < tmpView.d[dim]; ++i) {
                      copy(dim + 1, base, tmpInd);
                      tmpInd += tmpView.s[dim];
                    }
                  }
                };
                copy(0, 0, 0);
                nextSrc = tmpView.t;
                nextDims = tmpView.d;
                nextStride = tmpView.s;
                nextOffset = tmpView.o;
                workingIndex += 1;
              } else {
                throw new TypeError(`cannot index ndarray with ndarray of type ${dataTypeNames[view.t.t]}`);
              }
              continue;
            }
            throw new TypeError('ndarray index expired: ensure slices are used immediately after creation');
          }
          if (!part[0]) throw new SyntaxError('invalid syntax (empty slice)');
          if (part[0] == 'true' || part[0] == 'false') {
            nextDims.splice(workingIndex, 0, +(part[0] == 'true'));
            nextStride.splice(workingIndex, 0, 0);
            workingIndex++;
            continue;
          }
          let ind = +part[0];
          if (parts.length == 1 && isNaN(ind)) return target[key];
          if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0D ndarray');
          ind = fixInd(ind, nextDims.splice(workingIndex, 1)[0]);
          nextOffset += ind * nextStride.splice(workingIndex, 1)[0]
        } else if (part.length > 3) {
          throw new SyntaxError(`invalid slice ${key}`);
        } else {
          if (workingIndex >= nextDims.length) throw new TypeError('cannot slice 0D ndarray');
          let step = +(part[2] || 1);
          if (step == 0 || !Number.isInteger(step)) {
            throw new TypeError(`invalid step ${step}`);
          }
          const t = step || 1;
          const s = part[0]
            ? Math.min(Math.max(fixInd(+part[0], nextDims[workingIndex], 1), 0), nextDims[workingIndex])
            : (step < 0 ? nextDims[workingIndex] - 1 : 0)
          const e = part[1]
            ? Math.min(Math.max(fixInd(+part[1], nextDims[workingIndex], 1), 0), nextDims[workingIndex])
            : (step < 0 ? -1 : nextDims[workingIndex]);
          nextOffset += s * nextStride[workingIndex];
          nextDims[workingIndex] = Math.max(Math.floor((e - s) / t), 0)
          nextStride[workingIndex++] *= t;
        }
      }
      if (unbox && !nextDims.length) return nextSrc.b[nextOffset];
      return new NDView<T, Dims>(nextSrc, nextDims, nextStride, nextOffset);
    }

    return new Proxy(this, {
      get: (target, key) => get(target, key, 1),
      set: (target, key, value) => {
        if (typeof key == 'string' && key.includes(indexablePrefix)) {
          throw new TypeError(
            'setting values through a mask is unsupported: try calling .set() after indexing (will not affect original)'
          );
        }
        const val = (get(target, key) as NDView<T, Dims>);
        if (val && val[ndvInternals]) val.set(value);
        else target[key as string] = value;
        return true;
      }
    });
  }

  private get [ndvInternals]() {
    return this;
  }

  // calculate offset
  private c(ind: number[]) {
    let offset = this.o;
    for (let i = 0; i < ind.length; ++i) offset += ind[i] * this.s[i];
    return offset;
  }

  *[Symbol.iterator]() {
    const target = this[ndvInternals];
    if (!target.ndim) throw new TypeError('cannot iterate over scalar');
    if (target.ndim == 1) {
      for (let i = 0; i < target.d[0]; ++i) {
        yield target.t.b[target.o + i * target.s[0]] as NDViewChild<T, D>;
      }
    } else {
      const nextDims = target.d.slice(1), nextStride = target.s.slice(1);
      for (let i = 0; i < target.d[0]; ++i) {
        yield new NDView<T, Dims>(target.t, nextDims, nextStride, target.o + i * target.s[0]) as NDViewChild<T, D>;
      }
    }
  }

  /**
   * Copies data from another ndarray into the current view
   * @param value The view to copy data from
   */
  set(value: Broadcastable<AssignableType<T>>) {
    const [target, val] = broadcast(this, value);
    if (!isAssignable(target.t.t, val.t.t)) {
      throw new TypeError(`cannot assign to ndarray of type ${dataTypeNames[val.t.t]} to ${dataTypeNames[target.t.t]}; for unsafe conversion, use copy()`);
    }
    if (target.d.filter(v => v != 1).length != target.ndim) {
      const shape = (a: unknown): number[] => Array.isArray(a) ? [a.length, ...shape(a[0])]  : [];
      throw new TypeError(`cannot broadcast ndarray of shape (${value && value[ndvInternals] ? (value as NDView).d.join(', ') : shape(value)}) to (${target.d.join(', ')})`);
    }
    const set = (dim: number, ind: number, valInd: number) => {
      if (dim == target.d.length) target.t.b[ind] = val.t.b[valInd];
      else {
        if (dim == target.d.length - 1 && target.s[dim] == 1) {
          if (val.s[dim] == 1 && (target.t.b['set'] && val.t.b['subarray'])) {
            (target.t.b as Uint8Array).set((val.t.b as Uint8Array).subarray(valInd, valInd + target.d[dim]), ind);
            return;
          }
          if (!val.s[dim] && (target.t.b['fill'])) {
            (target.t.b as Uint8Array).fill(val.t.b[valInd] as number, ind, ind + target.d[dim]);
            return;
          }
        }
        for (let i = 0; i < target.d[dim]; ++i) {
          set(dim + 1, ind, valInd);
          ind += target.s[dim];
          valInd += val.s[dim];
        }
      }
    }
    set(0, target.o, val.o);
  }

  /**
   * Gets the value at a fully qualified index. For slicing, use the bracket notation.
   * @param index The index to get. Should have the same number of dimensions as the ndarray
   * @returns The value at the given index
   */
  get(index?: readonly number[]): IndexType<T>;
  /**
   * Gets the value at a fully qualified index. For slicing, use the bracket notation.
   * @param index The indices to get. Should be the same length as the same number of dimensions in
   *              the ndarray
   * @returns The value at the given index
   */
  get(...index: readonly number[]): IndexType<T>;
  get(...maybeIndex: unknown[]) {
    let index = (Array.isArray(maybeIndex[0]) ? maybeIndex[0] : maybeIndex) as number[];
    const target = this[ndvInternals];
    if (index.length != target.ndim) {
      throw new TypeError(`index of size ${index.length} cannot be used on ndarray with ${target.ndim} dimensions`);
    }
    let o = target.o;
    for (let i = 0; i < index.length; ++i) {
      o += fixInd(index[i], target.d[i]) * target.s[i];
    }
    return target.t.b[o];
  }

  /**
   * Converts the ndarray into a pretty representation
   * @returns A prettified string representing the ndarray
   */
  toString() {
    const target = this[ndvInternals];
    const stringify = target.t.t == DataType.Any || target.t.t == DataType.String
      ? (v: unknown) => JSON.stringify(v)
      : target.t.t == DataType.Float32 || target.t.t == DataType.Float64
        ? (v: number) => v.toString() + (Number.isInteger(v) ? '.' : '')
        : (v: unknown) => v.toString();
    if (!target.d.length) return stringify(target.t.b[target.o]);
    let maxLen = 0;
    let maxPre = 0;
    let maxPost = 0;
    let maxImagPre = 0;
    let maxImagPost = 0;
    const list = (dim: number, ind: number): RecursiveArray<string> => {
      if (dim == target.d.length) {
        let result = stringify(target.t.b[ind]);
        maxLen = Math.max(maxLen, result.length);
        if (target.t.t == DataType.Float32 || target.t.t == DataType.Float64) {
          maxPre = Math.max(maxPre, result.indexOf('.'));
          maxPost = Math.max(maxPost, result.length - result.indexOf('.'))
        } else if (target.t.t == DataType.Complex) {
          let midInd = result.indexOf('+');
          if (midInd == -1) midInd = result.indexOf('-');
          maxPre = Math.max(maxPre, result.indexOf('.'));
          maxPost = Math.max(maxPost, midInd - maxPre);
          let imag = result.slice(midInd);
          maxImagPre = Math.max(maxImagPre, imag.indexOf('.'));
          maxImagPost = Math.max(maxImagPost, imag.length - imag.indexOf('.'));
        }
        return result;
      }
      const s = target.s[dim], d = target.d[dim];
      if (d < 7) {
        let outs = [];
        for (let i = 0; i < d; ++i) {
          outs.push(list(dim + 1, ind));
          ind += s;
        }
        return outs;
      }
      return [
        list(dim + 1, ind),
        list(dim + 1, ind += s),
        list(dim + 1, ind += s),
        '...',
        list(dim + 1, ind += s * (d - 5)),
        list(dim + 1, ind += s),
        list(dim + 1, ind += s)
      ];
    }
    const result = list(0, target.o) as RecursiveArray<string>[];
    const applyPadding = target.t.t == DataType.Float32 || target.t.t == DataType.Float64
      ? (val: string) => val.slice(0, val.indexOf('.')).padStart(maxPre) + val.slice(val.indexOf('.')).padEnd(maxPost)
      : target.t.t == DataType.Complex
        ? (val: string) => {
          let midInd = val.indexOf('+');
          if (midInd == -1) midInd = val.indexOf('-');
          let real = val.slice(0, midInd);
          let imag = val.slice(midInd);
          return real.slice(0, real.indexOf('.')).padStart(maxPre) + real.slice(real.indexOf('.')).padEnd(maxPost) +
            imag.slice(0, imag.indexOf('.')).padStart(maxImagPre) + imag.slice(imag.indexOf('.')).padEnd(maxImagPost);
        }
        : target.t.t == DataType.Any
          ? (val: string) => val
          : (val: string) => val.padStart(maxLen);
    const concat = (arr: RecursiveArray<string>[], dim: number, indent: number) => 
      (arr as unknown as string) == '...'
        ? arr
        : `[${dim == target.d.length - 1
          ? arr.map(v => v == '...' ? v : applyPadding(v as string)).join(', ')
          : arr.map((val, i) => ' '.repeat(i && indent) +
              concat(val as RecursiveArray<string>[], dim + 1, indent + 1)
            ).join(dim == target.d.length - 2 ? ',\n' : ',\n\n')
        }]`;
    return `array(${concat(result, 0, 7)}, shape=(${this.d.join(', ')}), dtype=${dataTypeNames[this.dtype]})`;
  }

  /**
   * Ravels the ndarray into a flat representation, avoiding copying if possible
   * @returns A flattened version of the ndarray, possible viewing the same contiguous buffer as
   *          the source array
   */
  ravel() {
    return this.reshape([this.size] as [number]);
  }

  /**
   * Gets a flat buffer containing the contents of the ndarray. Often useful for usage within
   * another ndarray library (e.g. ONNX Runtime Web's Tensor) 
   * @returns A flat array containing the contents of this view, possibly pointing to the same
   *          buffer as the ndarray
   */
  toRaw(): DataTypeBuffer<T> {
    switch (this.t.t) {
      case DataType.String:
      case DataType.Any:
        return this.flatten().t.b;
      case DataType.Bool: {
        const raveled = this.ravel() as NDView<DataType.Bool>;
        return new Bitset(
          raveled.t.b.buffer,
          raveled.size,
          raveled.o + raveled.t.b.offset
        ) as DataTypeBuffer<T>;
      }
      case DataType.Complex: {
        const raveled = this.ravel() as NDView<DataType.Complex>;
        return new ComplexArray(
          raveled.t.b.buffer.subarray(raveled.o << 1, (raveled.o + raveled.size) << 1)
        ) as DataTypeBuffer<T>;
      }
      default: {
        const raveled = this.ravel();
        return (raveled.t.b as DataTypeBuffer<
          Exclude<DataType, DataType.Any | DataType.String | DataType.Bool | DataType.Complex>
        >).subarray(raveled.o, raveled.size) as DataTypeBuffer<T>;
      }
    }
  }

  [Symbol.for('nodejs.util.inspect.custom')](_: number, opts: { showProxy: boolean }) {
    if (opts.showProxy) {
      return this;
    }
    return this.toString();
  }

  [Symbol.toPrimitive](hint: 'number' | 'string' | 'default') {
    if (!this.ndim) {
      return hint == 'string' ? this.toString() : this.t.b[this.o];
    }
    if (hint == 'number') return NaN;
    const id = getFreeID();
    recentAccesses.set(id, this[ndvInternals]);
    queueMicrotask(() => recentAccesses.delete(id));
    return `${indexablePrefix}${zws.repeat(id)}<${dataTypeNames[this.t.t]}>(${this.d.join('x')}) [...]`;
  }

  /**
   * Reshapes an ndarray to new dimensions matching the original size in row-major order
   * @param dims The new dimensions to reshape into
   */
  reshape<ND extends Dims>(dims: ND): NDView<T, ND>;
  /**
   * Reshapes an ndarray to new dimensions matching the original size in row-major order
   * @param dims The new dimensions to reshape into
   */
  reshape<ND extends Dims>(...dims: ND): NDView<T, ND>;
  reshape<ND extends Dims>(...maybeDims: ND | [ND]): NDView<T, ND> {
    let dims = (Array.isArray(maybeDims[0]) ? maybeDims[0] : maybeDims) as ND;
    const target = this[ndvInternals], size = target.size;
    if (dims.some(v => !Number.isInteger(v))) {
      throw new TypeError(`cannot reshape to non-integral dimensions (${dims.join(', ')})`)
    }
    let calcSize = dims.reduce((a, b) => a * b, 1);
    let negInd = dims.findIndex(a => a < 0);
    if (negInd > -1) {
      if (dims.slice(negInd + 1).findIndex(a => a < 0) > -1) {
        throw new TypeError('can only specify one unknown dimension');
      }
      let spareDim = size * dims[negInd] / calcSize;
      if (Number.isInteger(spareDim)) {
        dims = dims.slice() as unknown as ND;
        (dims as unknown as number[])[negInd] = spareDim;
        calcSize = size;
      }
    }
    if (calcSize != size) {
      throw new TypeError(`dimensions (${dims.join(', ')}) do not match data length ${size}`);
    }
    const cd: number[] = [], cs: number[] = [], stride: number[] = dims.map(() => 0);
    for (let i = 0; i < target.ndim; ++i) {
      if (target.d[i] != 1) {
        cd.push(target.d[i]);
        cs.push(target.s[i]);
      }
    }
    let s = 0, e = 1, ns = 0, ne = 1;
    while (s < cd.length && ns < dims.length) {
      let srcChunks = cd[s], dstChunks = dims[ns];
      
      while (srcChunks != dstChunks) {
        if (srcChunks < dstChunks) srcChunks *= cd[e++];
        else dstChunks *= dims[ne++];
      }
      
      for (let i = s + 1; i < e; ++i) {
        if (cs[i - 1] != cd[i] * cs[i]) {
          return target.flatten().reshape(dims);
        }
      }
      
      stride[ne - 1] = cs[e - 1];
      for (let i = ne - 1; i > ns; --i) stride[i - 1] = stride[i] * dims[i];
      s = e++, ns = ne++;
    }
    return new NDView(target.t, dims, stride, target.o);
  }

  /**
   * Reverses the order of the axes in the ndarray
   * @returns The transposed ndarray as a view of the same buffer
   */
  transpose(): NDView<T, number[]>;
  /**
   * Reorders the axes in the ndarray
   * @param order The new axis order, where each element of the array corresponds to the current
   *              index of the axis in this ndarray's dimensions
   * @returns An ndarray with reordered axes as a view of the same buffer
   */
  transpose(order: readonly number[]): NDView<T, number[]>;
  /**
   * Reorders the axes in the ndarray
   * @param order The new axis order, where each value corresponds to the current index of the
   *              axis in this ndarray's dimensions
   * @returns An ndarray with reordered axes as a view of the same buffer
   */
  transpose(...order: readonly number[]): NDView<T, number[]>;
  transpose(...maybeOrder: unknown[]) {
    let order = (!maybeOrder.length || Array.isArray(maybeOrder[0]) ? maybeOrder[0] : maybeOrder) as Dims;
    if (!order) {
      order = this.d.map((_, i) => -i - 1);
    } else if (order.length != this.ndim) {
      throw new TypeError(`order length ${order.length} does not match data dimensions (${this.d.join(', ')})`);
    }
    const newDims: number[] = [];
    const newStrides: number[] = [];
    const seen = new Set<number>();
    for (let ord of order) {
      ord = fixInd(ord, this.ndim);
      if (seen.has(ord)) throw new TypeError(`repeated axis ${ord} in transpose`);
      seen.add(ord);
      newStrides.push(this.s[ord]);
      newDims.push(this.d[ord]);
    }
    return new NDView(this.t, newDims, newStrides, this.o);
  }

  /**
   * Flattens the ndarray into a single dimension. Similar to `ravel()` but always copies the data
   * @returns A copy of the array flattened into one dimension
   */
  flatten() {
    const target = this[ndvInternals];
    const ret = ndarray(target.t.t, [target.size] as [number]);
    const dst = ret[ndvInternals];
    let dstInd = -1;
    const set = (dim: number, srcInd: number) => {
      if (dim == target.d.length) {
        dst.t.b[++dstInd] = target.t.b[srcInd];
      } else {
        for (let i = 0; i < target.d[dim]; ++i) {
          set(dim + 1, srcInd);
          srcInd += target.s[dim];
        }
      }
    }
    set(0, target.o);
    return ret;
  }

  /**
   * Copies the contents of an ndarray of a different datatype into this one, even if the types are not directly assignable.
   * Prefer set() whenever possible (i.e. the types are guaranteed to be assignable).
   * @param from The array to copy from
   */
  copy(from: Broadcastable<DataType>) {
    const [target, val] = broadcast(this, from);
    const cast = target.t.t == DataType.Int64 || target.t.t == DataType.Uint64
      ? (v: unknown) => BigInt(+v)
      : target.t.t <= DataType.Float64
        ? (v: unknown) => Number(v)
        : (v: unknown) => v;
    const set = (dim: number, ti: number, oi: number) => {
      if (dim == target.d.length) {
        target.t.b[oi] = cast(val.t.b[ti]);
      } else {
        for (let i = 0; i < target.d[dim]; ++i) {
          set(dim + 1, ti, oi);
          ti += val.s[dim];
          oi += target.s[dim];
        }
      }
    }
    set(0, val.o, target.o);
  }

  /**
   * Copies an ndarray as a new datatype, even if the types are not directly assignable
   * @param dtype The datatype to cast to
   * @returns A new array with the given datatype
   */
  astype<NT extends DataType>(dtype: NT): NDView<NT, D> {
    const result = ndarray(dtype, this.d);
    result.copy(this);
    return result;
  }

  /**
   * The shape of the ndarray, e.g. [10, 20, 5]
   */
  get shape() {
    return this.d.slice() as Dims as D;
  }

  /**
   * The number of dimensions in the ndarray
   */
  get ndim() {
    return this.d.length;
  }

  /**
   * The total number of elements in the ndarray
   */
  get size() {
    return this.d.reduce((a, b) => a * b, 1);
  }

  /**
   * The ndarray's datatype
   */
  get dtype() {
    return this.t.t;
  }

  /**
   * The ndarray's transpose
   */
  get T() {
    return this.transpose(); 
  }
}