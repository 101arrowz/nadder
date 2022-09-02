import { DataType, Dims, ndarray, NDView } from '../core';
import { dataTypeNames } from '../core/datatype';

export const ndvInternals = Symbol('ndview-internals');
export type UnionToIntersection<U> =  (U extends unknown ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;
export const makeOut = <T extends DataType, S extends Dims>(name: string, shape: S, type: T, out?: unknown) => {
  if (out) {
    if (!out[ndvInternals]) {
      throw new TypeError(`matmul expected output to be an ndarray`);
    }
    if (out['t'].t != type) {
      throw new TypeError(`matmul expected output to have type ${dataTypeNames[type]} but got ${dataTypeNames[out['t'].t]}`);
    }
    if ((out as NDView).ndim != shape.length || out['d'].some((v, i) => shape[i] != v)) {
      throw new TypeError(`${name} expected output shape (${out['d'].join(', ')}) to match (${shape.join(', ')})`);
    }
  } else {
    out = ndarray(type, shape);
  }
  return out as NDView<T, S>;
}