import { Dims, ndarray, NDView } from '../../core/ndarray';
import { DataType, IndexType } from '../../core/datatype';
import { broadcast } from '../broadcast';
import { Bitset } from '../containers';

type MultiType = readonly DataType[];
type MultiTypeArgs = readonly MultiType[];

type AnyDataType<T extends MultiType> = IndexType<T[number]>;
type OpReturnType<T extends MultiType> = '1' extends keyof T ? [...({ [K in keyof T]: IndexType<T[K]> })] : IndexType<T[0]>;
type OpGetReturnArgs<T extends MultiTypeArgs> = [...({ [I in keyof T]: T[I][number] })];
type OpArgs<T extends MultiTypeArgs> = [...({ [I in keyof T]: IndexType<T[I][number]> })];
type OpImplFunction<T extends MultiTypeArgs, TR extends MultiType> = (...args: OpArgs<T>) => OpReturnType<TR>;
type OpImpl<T extends MultiTypeArgs, TR extends MultiType> = [args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>];

type UfuncReturnType<T extends MultiType, D extends Dims> = '1' extends keyof T ? [...({ [K in keyof T]: NDView<T[K], D> })] : NDView<T[0], D>;

type UfuncOpts<T extends MultiTypeArgs, TR extends MultiType, D extends Dims> = {
  out?: UfuncReturnType<TR, D>;
  where?: NDView<DataType.Bool, D>;
  dtype?: DataType;
};
type UfuncArgs<T extends MultiTypeArgs, TR extends MultiType, D extends Dims> =  [...args: ({ [I in keyof T]: NDView<T[I][number], D> | AnyDataType<T[I]> }), opts: UfuncOpts<T, TR, D> | void];
type UfuncSig<T> = T extends OpImpl<infer T, infer TR> ? (<D extends Dims>(...args: UfuncArgs<T, TR, D>) => UfuncReturnType<TR, D>) : never;
type UnionToIntersection<U> =  (U extends unknown ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
type Ufuncify<Tuple extends readonly unknown[]> = UnionToIntersection<{ [Index in keyof Tuple]: UfuncSig<Tuple[Index]> }[number]>;

export const opImpl = <T extends MultiTypeArgs, TR extends MultiType>(args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>): OpImpl<T, TR> =>
  [args, result, impl];

export const ufunc = <T extends readonly OpImpl<MultiTypeArgs, MultiType>[]>(name: string, nin: number, nout: number, ...impls: T): Ufuncify<T> => {
  const fastImpls = impls.map(([args, result, impl]) => [args.map(types => types.reduce((a, b) => a | b, 0)), result, impl] as const);
  // assertion: nin > 0, nout > 0, impls.every(([args, result, impl]) => args.length == nin && result.length == nout)
  return ((...args: UfuncArgs<MultiTypeArgs, MultiType, Dims>) => {
    if (args.length > nin + 1 || args.length < nin) throw new TypeError(`${name} takes ${nin} arguments and optional arguments; got ${args.length} arguments`);
    let { where = true, out, dtype } = (args.length > nin ? args.pop() : {}) as UfuncOpts<MultiTypeArgs, MultiType, Dims>;
    const [ndWhere, ...inputs] = broadcast(where, ...(args as NDView[]));
    const possibleImpls = fastImpls.filter(([ins]) => ins.every((mask, i) => mask & inputs[i]['t'].t));
    if (!possibleImpls.length) throw new TypeError(`${name} is not implemented for the given arguments`);
    const chosenImpl = possibleImpls.find(([_, outs]) => outs.every(t => t == dtype)) || possibleImpls[0];
    const [ins, outs, impl] = chosenImpl;
    const dims = inputs[0]['d'];
    if (nout > 1) {
      if (out) {
        if (!Array.isArray(out) || out.length != nout) {
          throw new TypeError(`${name} expects out to be an array of ${nout} outputs`);
        }
        for (let i = 0; i < out.length; ++i) {
          const curout = out[i] as NDView;
          if (curout['d'].length != dims.length || curout['d'].some((v, i) => dims[i] != v)) {
            throw new TypeError(`${name} expected broadcast shape (${dims.join(', ')}) to match output ${i + 1} shape (${curout['d'].join(', ')})`);
          }
        }
      } else out = outs.map(dtype => ndarray(dtype, dims)) as unknown as typeof out;

      for (const index of inputs[0]['r']()) {
        if (!ndWhere['t'].b[ndWhere['c'](index)]) continue;
        const values = inputs.map(input => input['t'].b[input['c'](index)]);
        const result = impl(...values);
        for (let i = 0; i < nout; ++i) {
          out[i]['t'].b[out[i]['c'](index)] = result[i];
        }
      }
    } else {
      if (out) {
        if (!(out instanceof NDView)) {
          throw new TypeError(`${name} expected out to be an ndarray`);
        }
        if (out['d'].length != dims.length || out['d'].some((v, i) => dims[i] != v)) {
          throw new TypeError(`${name} expected broadcast shape (${dims.join(', ')}) to match output shape (${out['d'].join(', ')})`);
        }
      } else out = ndarray(outs[0], dims);
      for (const index of inputs[0]['r']()) {
        if (!ndWhere['t'].b[ndWhere['c'](index)]) continue;
        const values = inputs.map(input => input['t'].b[input['c'](index)]);
        const result = impl(...values);
        out['t'].b[out['c'](index)] = result;
      }
      return out;
    }
  }) as Ufuncify<T>;
}