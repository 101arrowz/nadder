import { Dims, NDView } from '../../core/ndarray';
import { DataType, dataTypeNames, IndexType } from '../../core/datatype';
import { broadcast, Broadcastable } from '../broadcast';
import { fixInd, makeOut, ndvInternals, UnionToIntersection } from '../internal';
import { ndarray, RecursiveArray } from '../helpers';

type MultiType = readonly DataType[];
type MultiTypeArgs = readonly MultiType[];

type OpReturnType<T extends MultiType> = '1' extends keyof T ? [...({ [K in keyof T]: IndexType<T[K]> })] : IndexType<T[0]>;
type OpArgs<T extends MultiTypeArgs> = [...({ [I in keyof T]: IndexType<T[I][number]> })];
type OpImpl<T extends MultiTypeArgs, TR extends MultiType> = readonly [args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>];

type UfuncReturnType<T extends MultiType, TF extends DataType, D extends Dims> = '1' extends keyof T ? [...({ [K in keyof T]: NDView<DataType extends TF ? T[K] : TF, D> })] : NDView<DataType extends TF ? T[0] : TF, D>;

interface BaseUfuncOpts<TR extends MultiType, TF extends DataType, D extends Dims> {
  /**
   * The array(s) into which to write the output of the ufunc
   */
  out?: UfuncReturnType<TR, TF, D>;
  
  /**
   * The datatype to use for the output
   */
  dtype?: TF;
};

interface LocatableUfuncOpts<TR extends MultiType, TF extends DataType, D extends Dims> extends BaseUfuncOpts<TR, TF, D> {
  /**
   * A mask for the array that decides where the ufunc should be applied
   */
  where?: Broadcastable<DataType.Bool>;
};


/**
 * Options for `ufunc.reduce()`
 */
export interface ReduceUfuncOpts<TR extends DataType, TF extends DataType> extends LocatableUfuncOpts<[TR], TF, Dims> {
  /**
   * The axis or axes over which to reduce
   */
  axis?: number | number[];

  /**
   * Whether or not to preserve 1-length dimensions for reduced axes
   */
  keepdims?: boolean;
  
  /**
   * The initial value for the reduction
   */
  initial?: IndexType<DataType extends TF ? TR : TF>
};

/**
 * Options for `ufunc.accumulate()`
 */
export interface AccumulateUfuncOpts<TR extends DataType, TF extends DataType, D extends Dims> extends BaseUfuncOpts<[TR], TF, D> {
  /**
   * The axis over which to accumulate
   */
  axis?: number;
};

/**
 * Options for calling ufuncs
 */
export type UfuncOpts<T extends MultiTypeArgs, TR extends MultiType, TF extends DataType, D extends Dims> =
  | LocatableUfuncOpts<TR, TF, D>
  | UfuncReturnType<TR, TF, D>;

type UfuncArgs<T extends MultiTypeArgs, TR extends MultiType, TF extends DataType, D extends Dims> = [...args: ({ [I in keyof T]: NDView<T[I][number], D> | RecursiveArray<IndexType<T[I][number]>> }), opts: UfuncOpts<T, TR, TF, D> | void];
type UfuncSig<T> = T extends OpImpl<infer T, infer TR> ? (<D extends Dims, TF extends DataType>(...args: UfuncArgs<T, TR, TF, D>) => UfuncReturnType<TR, TF, D>) & ('1' extends keyof T ? '1' extends keyof TR ? unknown : {
  /**
   * Reduces an array with this operation across the supplied dimension(s)
   * @param target The ndarray to reduce
   * @param opts Options for the reduction
   */
  reduce<TF extends DataType>(target: Broadcastable<T[number][number]>, opts?: ReduceUfuncOpts<TR[0], TF>): UfuncReturnType<TR, TF, Dims>;
  
  /**
   * Accumulates the entries of an array with this operation across the supplied dimension
   * @param target The ndarray to accumulate
   * @param opts Options for the accumulation
   */
  accumulate<D extends Dims, TF extends DataType>(target: NDView<T[number][number], D> | RecursiveArray<IndexType<T[number][number]>>, opts?: AccumulateUfuncOpts<TR[0], TF, D>): UfuncReturnType<TR, TF, D>;
} : unknown) : never;
type Ufuncify<Tuple extends readonly unknown[], I> = UnionToIntersection<{ [Index in keyof Tuple]: UfuncSig<Tuple[Index]> }[number]> & (I extends never ? unknown : {
  identity: I
});

export const opImpl = <T extends MultiTypeArgs, TR extends MultiType>(args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>): OpImpl<T, TR> =>
  [args, result, impl];

// TODO: fix case of adding scalars in types

export const ufunc = <T extends readonly OpImpl<MultiTypeArgs, MultiType>[], I>(name: string, nin: number, nout: number, identity: I, ...impls: T): Ufuncify<T, I> => {
  const fastImpls = impls.map(([args, result, impl]) => [args.map(types => types.reduce((a, b) => a | b, 0)), result, impl] as const);
  const chooseImpl = (inputs: NDView[], dtype: DataType) => {
    const possibleImpls = fastImpls.filter(([ins]) => ins.every((mask, i) => mask & inputs[i]['t'].t));
    if (!possibleImpls.length) throw new TypeError(`${name} is not implemented for the given arguments`);
    const chosenImpl = dtype
      ? possibleImpls.find(([_, outs]) => outs.every(t => t == dtype)) || possibleImpls[0]
      : (dtype = possibleImpls[0][1][0], possibleImpls[0]);
    return {
      i: chosenImpl,
      d: dtype
    };
  }
  // assertion: nin > 0, nout > 0, impls.every(([args, result, impl]) => args.length == nin && result.length == nout)
  const fn = ((...args: UfuncArgs<MultiTypeArgs, MultiType, DataType, Dims>) => {
    if (args.length > nin + 1 || args.length < nin) throw new TypeError(`${name} takes ${nin} arguments and optional arguments; got ${args.length} arguments`);
    let opts = (args.length > nin && args.pop() || {}) as UfuncOpts<MultiTypeArgs, MultiType, DataType, Dims>;
    if (nout > 1 ? Array.isArray(opts) && opts.every(o => o && o[ndvInternals]) : opts[ndvInternals]) {
      opts = { out: opts as NDView };
    }
    let { where = true, out, dtype } = opts;
    const [ndWhere, ...inputs] = broadcast(where, ...(args as NDView[]));
    if (ndWhere['t'].t != DataType.Bool) throw new TypeError(`${name} expects where to be a boolean ndarray`);
    const choice = chooseImpl(inputs, dtype);
    dtype = choice.d;
    const [ins, outs, impl] = choice.i;
    const dims = inputs[0]['d'];
    if (nout > 1) {
      if (out) {
        if (!Array.isArray(out) || out.length != nout) {
          throw new TypeError(`${name} expects out to be an array of ${nout} outputs`);
        }
        for (let i = 0; i < out.length; ++i) {
          const curout = out[i] as NDView;
          if (!curout || !curout[ndvInternals]) {
            throw new TypeError(`${name} expected output ${i + 1} to be an ndarray`);
          }
          if (curout['t'].t != dtype) {
            throw new TypeError(`${name} expected output ${i + 1} to have type ${dataTypeNames[dtype]} but got ${dataTypeNames[curout['t'].t]}`);
          }
          if (curout.ndim != dims.length || curout['d'].some((v, i) => dims[i] != v)) {
            throw new TypeError(`${name} expected broadcast shape (${dims.join(', ')}) to match output ${i + 1} shape (${curout['d'].join(', ')})`);
          }
        }
      } else out = outs.map(() => ndarray(dtype, dims)) as unknown as typeof out;
      const assign = (out as unknown as NDView[]).map(v => v[ndvInternals]);
      const coord = dims.map(() => -1);
      // TODO: fastpaths
      const call = (dim: number) => {
        if (dim == dims.length) {
          if (ndWhere['t'].b[ndWhere['c'](coord)]) {
            const values = inputs.map(input => input['t'].b[input['c'](coord)]);
            const result = impl(...values);
            for (let i = 0; i < nout; ++i) {
              assign[i]['t'].b[assign[i]['c'](coord)] = result[i];
            }
          }
        } else {
          for (coord[dim] = 0; coord[dim] < dims[dim]; ++coord[dim]) {
            call(dim + 1);
          }
        }
      }
      call(0);
      return dims.length ? out : (out as unknown as NDView[]).map(o => o['t'].b[o['o']]);
    } else {
      out = makeOut(name, dims, dtype, out);
      const assign = out[ndvInternals];
      if (nin == 2) {
        const [in0, in1] = inputs;
        const callWhere = (dim: number, ind0: number, ind1: number, outInd: number, whereInd: number) => {
          if (dim == dims.length) {
            if (ndWhere['t'].b[whereInd]) {
              assign['t'].b[outInd] = impl(in0['t'].b[ind0], in1['t'].b[ind1]);
            }
          } else {
            for (let i = 0; i < dims[dim]; ++i) {
              callWhere(dim + 1, ind0, ind1, outInd, whereInd);
              ind0 += in0['s'][dim];
              ind1 += in1['s'][dim];
              outInd += assign['s'][dim];
              whereInd += ndWhere['s'][dim];
            }
          }
        };
        const slowCall = (dim: number, ind0: number, ind1: number, outInd: number) => {
          if (dim == dims.length - 4) call(dim, ind0, ind1, outInd)
          else {
            for (let i = 0; i < dims[dim]; ++i) {
              slowCall(dim + 1, ind0, ind1, outInd);
              ind0 += in0['s'][dim];
              ind1 += in1['s'][dim];
              outInd += assign['s'][dim];
            }
          }
        };
        const call = (dim: number, ind0: number, ind1: number, outInd: number) => {
          const left = dims.length - dim;
          if (left > 4) return slowCall(dim, ind0, ind1, outInd);
          const i0b = in0['t'].b, i1b = in1['t'].b, ob = assign['t'].b;
          const i0s = in0['s'].slice(dim), i1s = in1['s'].slice(dim), os = assign['s'].slice(dim);
          if (left) {
            const id = dims[dim];
            let i0i = i0s[0], i1i = i1s[0], oi = os[0];
            if (left > 1) {
              const jd = dims[dim + 1];
              let i0j = i0s[1], i1j = i1s[1], oj = os[1];
              i0i -= i0j * jd;
              i1i -= i1j * jd;
              oi -= oj * jd;
              if (left > 2) {
                const kd = dims[dim + 2];
                let i0k = i0s[2], i1k = i1s[2], ok = i1s[2];
                i0j -= i0k * kd;
                i1j -= i1k * kd;
                oj -= ok * kd;
                if (left > 3) {
                  const ld = dims[dim + 3];
                  let i0l = i0s[3], i1l = i1s[3], ol = i1s[3];
                  i0k -= i0l * ld;
                  i1k -= i1l * ld;
                  ok -= ol * ld;
                  for (let i = 0; i < id; ++i) {
                    for (let j = 0; j < jd; ++j) {
                      for (let k = 0; k < kd; ++k) {
                        for (let l = 0; l < ld; ++l) {
                          ob[outInd] = impl(i0b[ind0], i1b[ind1]);
                          ind0 += i0l;
                          ind1 += i1l;
                          outInd += ol;
                        }
                        ind0 += i0k;
                        ind1 += i1k;
                        outInd += ok;
                      }
                      ind0 += i0j;
                      ind1 += i1j;
                      outInd += oj;
                    }
                    ind0 += i0i;
                    ind1 += i1i;
                    outInd += oi;
                  }
                } else {
                  for (let i = 0; i < id; ++i) {
                    for (let j = 0; j < jd; ++j) {
                      for (let k = 0; k < kd; ++k) {
                        ob[outInd] = impl(i0b[ind0], i1b[ind1]);
                        ind0 += i0k;
                        ind1 += i1k;
                        outInd += ok;
                      }
                      ind0 += i0j;
                      ind1 += i1j;
                      outInd += oj;
                    }
                    ind0 += i0i;
                    ind1 += i1i;
                    outInd += oi;
                  }
                }
              } else {
                for (let i = 0; i < id; ++i) {
                  for (let j = 0; j < jd; ++j) {
                    ob[outInd] = impl(i0b[ind0], i1b[ind1]);
                    ind0 += i0j;
                    ind1 += i1j;
                    outInd += oj;
                  }
                  ind0 += i0i;
                  ind1 += i1i;
                  outInd += oi;
                }
              }
            } else {
              for (let i = 0; i < id; ++i) {
                ob[outInd] = impl(i0b[ind0], i1b[ind1]);
                ind0 += i0i;
                ind1 += i1i;
                outInd += oi;
              }
            }
          } else {
            ob[outInd] = impl(i0b[ind0], i1b[ind1])
          }
        }
        (where === true ? call : callWhere)(0, in0['o'], in1['o'], assign['o'], ndWhere['o']);
      } else if (nin == 1) {
        const [input] = inputs;
        const callWhere = (dim: number, ind: number, outInd: number, whereInd: number) => {
          if (dim == dims.length) {
            if (ndWhere['t'].b[whereInd]) {
              assign['t'].b[outInd] = impl(input['t'].b[ind]);
            }
          } else {
            for (let i = 0; i < dims[dim]; ++i) {
              callWhere(dim + 1, ind, outInd, whereInd);
              ind += input['s'][dim];
              outInd += assign['s'][dim];
              whereInd += ndWhere['s'][dim];
            }
          }
        };
        const call = (dim: number, ind: number, outInd: number) => {
          if (dim == dims.length) {
            assign['t'].b[outInd] = impl(input['t'].b[ind]);
          } else {
            for (let i = 0; i < dims[dim]; ++i) {
              call(dim + 1, ind, outInd);
              ind += input['s'][dim];
              outInd += assign['s'][dim];
            }
          }
        };
        (where === true ? call : callWhere)(0, input['o'], assign['o'], ndWhere['o']);
      } else {
        const coord = dims.map(() => -1);
        // TODO: fastpaths
        const call = (dim: number) => {
          if (dim == dims.length) {
            if (ndWhere['t'].b[ndWhere['c'](coord)]) {
              const values = inputs.map(input => input['t'].b[input['c'](coord)]);
              assign['t'].b[assign['c'](coord)] = impl(...values);
            }
          } else {
            for (coord[dim] = 0; coord[dim] < dims[dim]; ++coord[dim]) {
              call(dim + 1);
            }
          }
        }
        call(0);
      }
      return dims.length ? out : out['t'].b[out['o']];
    }
  })
  Object.defineProperty(fn, 'reduce', {
    value: (in0: NDView, opts: ReduceUfuncOpts<DataType, DataType>) => {
      if (nin != 2 || nout != 1) {
        throw new TypeError('can only reduce over binary functions');
      }
      let { where = true, out, dtype, axis = 0, keepdims, initial = identity } = opts || {};
      const [ndWhere, target] = broadcast(where, in0);
      if (ndWhere['t'].t != DataType.Bool) throw new TypeError(`${name} expects where to be a boolean ndarray`);
      const axes = axis == null
        ? target['d'].map((_, i) => i)
        : (Array.isArray(axis)
          ? axis
          : [axis]).map(v => fixInd(v, target.ndim));
      if ((new Set(axes)).size != axes.length) {
        throw new TypeError('duplicate axes for reduce');
      }
      if (identity == null && axes.length != 1) {
        throw new TypeError(`${name} undefined over multiple axes`);
      }
      axes.sort((a, b) => b - a);
      if (!dtype) {
        dtype = out && out[ndvInternals] && out.dtype || target.dtype;
      }
      const outDims = keepdims
        ? target['d'].map((d, i) => axes.includes(i) ? 1 : d)
        : target['d'].filter((_, i) => !axes.includes(i))
      out = makeOut(name, outDims, dtype, out);
      const assign = out[ndvInternals];
      const endShape = target['d'].slice(axes[0] + 1);
      const endWhere = new NDView(
        ndWhere['t'],
        endShape,
        ndWhere['s'].slice(axes[0] + 1),
        0
      );
      const wv = endWhere[ndvInternals];
      const endIn = new NDView(
        target['t'],
        endShape,
        target['s'].slice(axes[0] + 1),
        0
      );
      const iv = endIn[ndvInternals];
      const endOut = new NDView(
        assign['t'],
        endShape,
        assign['s'].slice(axes[0] - axes.length + 1),
        0
      );
      const ov = endOut[ndvInternals];
      if (initial == null) {
        if (!target.size) {
          throw new TypeError(`cannot reduce with ${name} over zero-size array without initial value`);
        }
        const axis = axes[0];
        const red = (dim: number, ind: number, outInd: number, whereInd: number) => {
          if (dim == axis) {
            ov['o'] = outInd;
            iv['o'] = ind;
            wv['o'] = whereInd;
            endOut.copy(endIn);
            for (let i = 1; i < target['d'][dim]; ++i) {
              iv['o'] += target['s'][dim];
              wv['o'] += ndWhere['s'][dim];
              (fn as (...args: unknown[]) => {})(
                endOut,
                endIn,
                { where: where === true || endWhere, out: endOut }
              );
            }
          } else {
            for (let i = 0; i < target['d'][dim]; ++i) {
              red(dim + 1, ind, outInd, whereInd);
              ind += target['s'][dim];
              outInd += assign['s'][dim];
              whereInd += ndWhere['s'][dim];
            }
          }
        }
        red(0, target['o'], assign['o'], ndWhere['o']);
      } else {
        out.set(initial);
        const redWhere = (dim: number, sub: number, ind: number, outInd: number, whereInd: number) => {
          if (axes.includes(dim)) {
            if (dim == axes[0]) {
              ov['o'] = outInd;
              iv['o'] = ind;
              wv['o'] = whereInd;
              for (let i = 0; i < target['d'][dim]; ++i) {
                (fn as (...args: unknown[]) => {})(
                  endOut,
                  endIn,
                  { where: endWhere, out: endOut }
                );
                iv['o'] += target['s'][dim];
                wv['o'] += ndWhere['s'][dim];
              }
            } else {
              for (let i = 0; i < target['d'][dim]; ++i) {
                redWhere(dim + 1, sub + 1, ind, outInd, whereInd);
                ind += target['s'][dim];
                whereInd += ndWhere['s'][dim];
              }
            }
          } else {
            for (let i = 0; i < target['d'][dim]; ++i) {
              redWhere(dim + 1, sub, ind, outInd, whereInd);
              ind += target['s'][dim];
              outInd += assign['s'][dim - sub];
              whereInd += ndWhere['s'][dim];
            }
          }
        }
        const red = (dim: number, sub: number, ind: number, outInd: number) => {
          if (axes.includes(dim)) {
            if (dim == axes[0]) {
              ov['o'] = outInd;
              iv['o'] = ind;
              for (let i = 0; i < target['d'][dim]; ++i) {
                (fn as (...args: unknown[]) => {})(
                  endOut,
                  endIn,
                  { out: endOut }
                );
                iv['o'] += target['s'][dim];
              }
            } else {
              for (let i = 0; i < target['d'][dim]; ++i) {
                red(dim + 1, sub + 1, ind, outInd);
                ind += target['s'][dim];
              }
            }
          } else {
            for (let i = 0; i < target['d'][dim]; ++i) {
              red(dim + 1, sub, ind, outInd);
              ind += target['s'][dim];
              outInd += assign['s'][dim - sub];
            }
          }
        }
        (where === true ? red : redWhere)(0, 0, target['o'], assign['o'], ndWhere['o']);
      }
      return outDims.length ? out : assign['t'].b[assign['o']];
    }
  });
  Object.defineProperty(fn, 'accumulate', {
    value: (in0: NDView, opts: AccumulateUfuncOpts<DataType, DataType, Dims>) => {
      if (nin != 2 || nout != 1) {
        throw new TypeError('can only reduce over binary functions');
      }
      const [target] = broadcast(in0);
      let { out, dtype, axis = 0 } = opts || {};
      axis = fixInd(axis, target.ndim);
      if (!dtype) {
        dtype = out && out[ndvInternals] && out.dtype || target.dtype;
      }
      const dims = target.shape;
      out = makeOut(name, dims, dtype, out);
      if (!out.size) return out;
      const assign = out[ndvInternals];
      const endShape = target['d'].slice(axis + 1);
      const endIn = new NDView(
        target['t'],
        endShape,
        target['s'].slice(axis + 1),
        0
      );
      const iv = endIn[ndvInternals];
      const endOut = new NDView(
        assign['t'],
        endShape,
        assign['s'].slice(axis + 1),
        0
      );
      const ov = endOut[ndvInternals];
      const endPrevOut = new NDView(
        assign['t'],
        endShape,
        assign['s'].slice(axis + 1),
        0
      );
      const pv = endPrevOut[ndvInternals];
      const acc = (dim: number, ind: number, outInd: number) => {
        if (dim == axis) {
          ov['o'] = pv['o'] = outInd;
          iv['o'] = ind;
          endPrevOut.copy(endIn);
          for (let i = 1; i < target['d'][dim]; ++i) {
            iv['o'] += target['s'][dim];
            pv['o'] = ov['o'];
            ov['o'] += assign['s'][dim];
            (fn as (...args: unknown[]) => {})(
              endPrevOut,
              endIn,
              { out: endOut }
            );
          }
        } else {
          for (let i = 0; i < target['d'][dim]; ++i) {
            acc(dim + 1, ind, outInd);
            ind += target['s'][dim];
            outInd += assign['s'][dim];
          }
        }
      }
      acc(0, target['o'], assign['o']);
      return out;
    }
  });
  return fn as Ufuncify<T, I>;
}