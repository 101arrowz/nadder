import { Dims, ndarray, NDView } from '../../core/ndarray';
import { DataType, IndexType } from '../../core/datatype';
import { broadcast } from '../broadcast';

type MultiType = readonly DataType[];
type MultiTypeArgs = readonly MultiType[];

type AnyDataType<T extends MultiType> = IndexType<{ [K in keyof T]: T[K] }[number]>;
type OpArgs<T extends MultiTypeArgs> = [...({ [I in keyof T]: AnyDataType<T[I]> })];
type OpImpl<T extends MultiTypeArgs, TR extends DataType> = [args: T, result: TR, impl: (...args: OpArgs<T>) => IndexType<TR>];

type UfuncArgs<T extends MultiTypeArgs, TR extends DataType, D extends Dims> =  [...args: ({ [I in keyof T]: NDView<T[I][number], D> }), dest: NDView<TR, D> | void];
type UfuncSig<T> = T extends OpImpl<infer T, infer TR> ? (<D extends Dims>(...args: UfuncArgs<T, TR, D>) => NDView<TR, D>) : never;
type UnionToIntersection<U> =  (U extends unknown ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
type Ufuncify<Tuple extends readonly unknown[]> = UnionToIntersection<{ [Index in keyof Tuple]: UfuncSig<Tuple[Index]> }[number]>;

export const opImpl = <T extends MultiTypeArgs, TR extends DataType>(args: T, result: TR, impl: (...args: OpArgs<T>) => IndexType<TR>): OpImpl<T, TR> =>
  [args, result, impl];


export const ufunc = <T extends readonly OpImpl<MultiTypeArgs, DataType>[]>(...impls: T): Ufuncify<T> => {

}

const add2 = ufunc(
  opImpl([[DataType.Int8, DataType.Int16], [DataType.Int8]] as const, DataType.Int8, (a, b) => a + b),
  opImpl([[DataType.Int8], [DataType.Int8]] as const, DataType.Float32, (a, b) => a + b)
);