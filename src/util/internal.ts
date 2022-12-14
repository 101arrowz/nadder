export const ndvInternals = Symbol('ndview-internals');
export type UnionToIntersection<U> =  (U extends unknown ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never;

import { Dims, NDView } from '../core/ndarray';
import { DataType, dataTypeNames } from '../core/datatype';
import { ndarray } from './helpers';

export const makeOut = <T extends DataType, S extends Dims>(name: string, shape: S, type: T, out?: unknown) => {
  if (out) {
    if (!out[ndvInternals]) {
      throw new TypeError(`${name} expected output to be an ndarray`);
    }
    if (out['t'].t != type) {
      throw new TypeError(`${name} expected output to have type ${dataTypeNames[type]} but got ${dataTypeNames[out['t'].t]}`);
    }
    if ((out as NDView).ndim != shape.length || out['d'].some((v, i) => shape[i] != v)) {
      throw new TypeError(`${name} expected output shape (${out['d'].join(', ')}) to match (${shape.join(', ')})`);
    }
  } else {
    out = ndarray(type, shape);
  }
  return out as NDView<T, S>;
}

export const fixInd = (ind: number, size: number, loose?: 1) => {
  if (!Number.isInteger(ind) || (!loose && (ind < -size || ind >= size))) {
    throw new RangeError(`index ${ind} out of bounds in dimension of length ${size}`);
  }
  return ind < 0 ? ind + size : ind;
}