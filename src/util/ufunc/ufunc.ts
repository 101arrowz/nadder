import { Dims, ndarray, NDView, RecursiveArray } from '../../core/ndarray';
import { DataType, dataTypeNames, IndexType } from '../../core/datatype';
import { broadcast } from '../broadcast';
import { ndvInternals } from '../internal';

type MultiType = readonly DataType[];
type MultiTypeArgs = readonly MultiType[];

type OpReturnType<T extends MultiType> = '1' extends keyof T ? [...({ [K in keyof T]: IndexType<T[K]> })] : IndexType<T[0]>;
type OpArgs<T extends MultiTypeArgs> = [...({ [I in keyof T]: IndexType<T[I][number]> })];
type OpImpl<T extends MultiTypeArgs, TR extends MultiType> = readonly [args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>];

type UfuncReturnType<T extends MultiType, TF extends DataType, D extends Dims> = '1' extends keyof T ? [...({ [K in keyof T]: NDView<DataType extends TF ? T[K] : TF, D> })] : NDView<DataType extends TF ? T[0] : TF, D>;

type UfuncOpts<T extends MultiTypeArgs, TR extends MultiType, TF extends DataType, D extends Dims> = {
  out?: UfuncReturnType<TR, TF, D>;
  where?: NDView<DataType.Bool, D> | boolean;
  dtype?: TF;
} | UfuncReturnType<TR, TF, D>;
type UfuncArgs<T extends MultiTypeArgs, TR extends MultiType, TF extends DataType, D extends Dims> =  [...args: ({ [I in keyof T]: NDView<T[I][number], D> | RecursiveArray<IndexType<T[I][number]>> }), opts: UfuncOpts<T, TR, TF, D> | void];
type UfuncSig<T> = T extends OpImpl<infer T, infer TR> ? (<D extends Dims, TF extends DataType>(...args: UfuncArgs<T, TR, TF, D>) => UfuncReturnType<TR, TF, D>) : never;
type UnionToIntersection<U> =  (U extends unknown ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
type Ufuncify<Tuple extends readonly unknown[]> = UnionToIntersection<{ [Index in keyof Tuple]: UfuncSig<Tuple[Index]> }[number]>;

export const opImpl = <T extends MultiTypeArgs, TR extends MultiType>(args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>): OpImpl<T, TR> =>
  [args, result, impl];

// TODO: fix case of adding scalars in types

export const ufunc = <T extends readonly OpImpl<MultiTypeArgs, MultiType>[]>(name: string, nin: number, nout: number, ...impls: T): Ufuncify<T> => {
  const fastImpls = impls.map(([args, result, impl]) => [args.map(types => types.reduce((a, b) => a | b, 0)), result, impl] as const);
  // assertion: nin > 0, nout > 0, impls.every(([args, result, impl]) => args.length == nin && result.length == nout)
  return ((...args: UfuncArgs<MultiTypeArgs, MultiType, DataType, Dims>) => {
    if (args.length > nin + 1 || args.length < nin) throw new TypeError(`${name} takes ${nin} arguments and optional arguments; got ${args.length} arguments`);
    let opts = (args.length > nin ? args.pop() : {}) as UfuncOpts<MultiTypeArgs, MultiType, DataType, Dims>;
    if (nout > 1 ? Array.isArray(opts) && opts.every(o => o[ndvInternals]) : opts[ndvInternals]) {
      opts = { out: opts as NDView };
    }
    let { where = true, out, dtype } = opts;
    const [ndWhere, ...inputs] = broadcast(where, ...(args as NDView[]));
    if (ndWhere['t'].t != DataType.Bool) throw new TypeError(`${name} expects where to be a boolean ndarray`);
    const possibleImpls = fastImpls.filter(([ins]) => ins.every((mask, i) => mask & inputs[i]['t'].t));
    if (!possibleImpls.length) throw new TypeError(`${name} is not implemented for the given arguments`);
    const chosenImpl = dtype
      ? possibleImpls.find(([_, outs]) => outs.every(t => t == dtype)) || possibleImpls[0]
      : (dtype = possibleImpls[0][1][0], possibleImpls[0]);
    const [ins, outs, impl] = chosenImpl;
    const dims = inputs[0]['d'];
    if (nout > 1) {
      if (out) {
        if (!Array.isArray(out) || out.length != nout) {
          throw new TypeError(`${name} expects out to be an array of ${nout} outputs`);
        }
        for (let i = 0; i < out.length; ++i) {
          const curout = out[i] as NDView;
          if (!curout[ndvInternals]) {
            throw new TypeError(`${name} expected output ${i + 1} to be an ndarray`);
          }
          if (curout['t'].t != dtype) {
            throw new TypeError(`${name} expected output ${i + 1} to have type ${dataTypeNames[dtype]} but got ${dataTypeNames[curout['t'].t]}`);
          }
          if (curout['d'].length != dims.length || curout['d'].some((v, i) => dims[i] != v)) {
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
      if (out) {
        if (!out[ndvInternals]) {
          throw new TypeError(`${name} expected out to be an ndarray`);
        }
        if (out['t'].t != dtype) {
          throw new TypeError(`${name} expected out to have type ${dataTypeNames[dtype]} but got ${dataTypeNames[out['t'].t]}`);
        }
        if (out['d'].length != dims.length || out['d'].some((v, i) => dims[i] != v)) {
          throw new TypeError(`${name} expected broadcast shape (${dims.join(', ')}) to match output shape (${out['d'].join(', ')})`);
        }
      } else out = ndarray(dtype, dims);
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
        const call = (dim: number, ind0: number, ind1: number, outInd: number) => {
          if (dim == dims.length) {
            assign['t'].b[outInd] = impl(in0['t'].b[ind0], in1['t'].b[ind1]);
          } else {
            for (let i = 0; i < dims[dim]; ++i) {
              call(dim + 1, ind0, ind1, outInd);
              ind0 += in0['s'][dim];
              ind1 += in1['s'][dim];
              outInd += assign['s'][dim];
            }
          }
        };
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
  }) as Ufuncify<T>;
}