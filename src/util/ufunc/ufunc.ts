import { Dims, ndarray, NDView } from '../../core/ndarray';
import { DataType, dataTypeNames, IndexType } from '../../core/datatype';
import { broadcast } from '../broadcast';
import { Bitset } from '../containers';

type MultiType = readonly DataType[];
type MultiTypeArgs = readonly MultiType[];

type AnyDataType<T extends MultiType> = IndexType<T[number]>;
type OpReturnType<T extends MultiType> = '1' extends keyof T ? [...({ [K in keyof T]: IndexType<T[K]> })] : IndexType<T[0]>;
type OpArgs<T extends MultiTypeArgs> = [...({ [I in keyof T]: IndexType<T[I][number]> })];
type OpImpl<T extends MultiTypeArgs, TR extends MultiType> = [args: T, result: TR, impl: (...args: OpArgs<T>) => OpReturnType<TR>];

type UfuncReturnType<T extends MultiType, TF extends DataType, D extends Dims> = '1' extends keyof T ? [...({ [K in keyof T]: NDView<DataType extends TF ? T[K] : TF, D> })] : NDView<DataType extends TF ? T[0] : TF, D>;

type UfuncOpts<T extends MultiTypeArgs, TR extends MultiType, TF extends DataType, D extends Dims> = {
  out?: UfuncReturnType<TR, TF, D>;
  where?: NDView<DataType.Bool, D> | boolean;
  dtype?: TF;
};
type UfuncArgs<T extends MultiTypeArgs, TR extends MultiType, TF extends DataType, D extends Dims> =  [...args: ({ [I in keyof T]: NDView<T[I][number], D> | AnyDataType<T[I]> }), opts: UfuncOpts<T, TR, TF, D> | void];
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
    let { where = true, out, dtype } = (args.length > nin ? args.pop() : {}) as UfuncOpts<MultiTypeArgs, MultiType, DataType, Dims>;
    const [ndWhere, ...inputs] = broadcast(where, ...(args as NDView[]));
    const possibleImpls = fastImpls.filter(([ins]) => ins.every((mask, i) => mask & inputs[i]['t'].t));
    if (!possibleImpls.length) throw new TypeError(`${name} is not implemented for the given arguments`);
    const chosenImpl = possibleImpls.find(([_, outs]) => outs.every(t => t == dtype)) || (dtype = dtype || possibleImpls[0][1][0], possibleImpls[0]);
    const [ins, outs, impl] = chosenImpl;
    const dims = inputs[0]['d'];
    if (nout > 1) {
      if (out) {
        if (!Array.isArray(out) || out.length != nout) {
          throw new TypeError(`${name} expects out to be an array of ${nout} outputs`);
        }
        for (let i = 0; i < out.length; ++i) {
          const curout = out[i] as NDView;
          if (!(curout instanceof NDView)) {
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

      for (const index of inputs[0]['r']()) {
        if (!ndWhere['t'].b[ndWhere['c'](index)]) continue;
        const values = inputs.map(input => input['t'].b[input['c'](index)]);
        const result = impl(...values);
        for (let i = 0; i < nout; ++i) {
          out[i]['t'].b[out[i]['c'](index)] = result[i];
        }
      }
      return dims.length ? out : (out as unknown as NDView[]).map(o => o['t'].b[o['o']]);
    } else {
      if (out) {
        if (!(out instanceof NDView)) {
          throw new TypeError(`${name} expected out to be an ndarray`);
        }
        if (out['t'].t != dtype) {
          throw new TypeError(`${name} expected out to have type ${dataTypeNames[dtype]} but got ${dataTypeNames[out['t'].t]}`);
        }
        if (out['d'].length != dims.length || out['d'].some((v, i) => dims[i] != v)) {
          throw new TypeError(`${name} expected broadcast shape (${dims.join(', ')}) to match output shape (${out['d'].join(', ')})`);
        }
      } else out = ndarray(dtype, dims);
      for (const index of inputs[0]['r']()) {
        if (!ndWhere['t'].b[ndWhere['c'](index)]) continue;
        const values = inputs.map(input => input['t'].b[input['c'](index)]);
        const result = impl(...values);
        out['t'].b[out['c'](index)] = result;
      }
      return dims.length ? out : out['t'].b[out['o']];
    }
  }) as Ufuncify<T>;
}