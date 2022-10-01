
import { Dims, NDView } from '../core/ndarray';
import { FlatArray } from '../core/flatarray';
import { ndvInternals } from './internal';
import {
  DataTypeBuffer, DataType, InferDataType, NumericType, BigNumericType, guessType, bestGuess
} from '../core/datatype';
import { Complex } from './types';

/**
 * Creates an empty ndarray from its type and dimensions
 * @param type The datatype for the ndarray
 * @param dimensions The dimensions of the ndarray
 * @returns An ndarray with the given dimensions and default values for the given datatype
 */
export function ndarray<T extends DataType, D extends Dims>(type: T, dimensions: D): NDView<T, D>;
/**
 * Wraps a flat buffer in an ndarray interface
 * @param data The buffer to wrap, with a length that matches the given dimensions
 * @param dimensions The dimensions of the ndarray
 * @returns An ndarray with the given dimensions and data
 */
export function ndarray<T extends DataTypeBuffer<DataType>, D extends Dims>(data: T, dimensions: D): NDView<InferDataType<T>, D>;
export function ndarray<T extends DataType, D extends Dims>(dataOrType: T | DataTypeBuffer<T>, dimensions: D) {
  if (dimensions.some(d => !Number.isInteger(d))) {
    throw new TypeError(`cannot reshape to non-integral dimensions (${dimensions.join(', ')})`)
  }
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

const recurseFind = (data: RecursiveArray<unknown>): [number[], DataType] => {
  if (Array.isArray(data)) {
    if (data.length == 0) return [[0], DataType.Any];
    const results = data.map(recurseFind);
    const newType = bestGuess(results.map(([, t]) => t));
    if (results.some(([dim]) => dim.length != results[0][0].length || dim.some((v, i) => results[0][0][i] != v))) {
      throw new TypeError('jagged ndarrays are not supported');
    }
    return [[data.length, ...results[0][0]], newType];
  }
  return [[], guessType(data)];
}

/**
 * A recursive list interface for use with `array()`
 */
export type RecursiveArray<T> = T | RecursiveArray<T>[];

/**
 * Creates an ndarray from nested Array objects
 * @param data The N-dimensional list to load data from
 * @returns An ndarray with the provided data represented in an efficient format
 */
export function array(data: RecursiveArray<number>): NDView<DataType.Int32 | DataType.Float64>;
/**
 * Creates an ndarray from nested Array objects
 * @param data The N-dimensional list to load data from
 * @returns An ndarray with the provided data represented in an efficient format
 */
export function array(data: RecursiveArray<bigint>): NDView<DataType.Int64>;
/**
 * Creates an ndarray from nested Array objects
 * @param data The N-dimensional list to load data from
 * @returns An ndarray with the provided data represented in an efficient format
 */
export function array(data: RecursiveArray<string>): NDView<DataType.String>;
/**
 * Creates an ndarray from nested Array objects
 * @param data The N-dimensional list to load data from
 * @returns An ndarray with the provided data represented in an efficient format
 */
export function array(data: RecursiveArray<boolean>): NDView<DataType.Bool>;
/**
 * Creates an ndarray from nested Array objects
 * @param data The N-dimensional list to load data from
 * @returns An ndarray with the provided data represented in an efficient format
 */
export function array(data: RecursiveArray<Complex>): NDView<DataType.Complex>;
/**
 * Creates an ndarray from nested Array objects
 * @param data The N-dimensional list to load data from
 * @returns An ndarray with the provided data represented in an efficient format
 */
export function array(data: RecursiveArray<unknown>): NDView<DataType.Any>;
export function array(data: RecursiveArray<unknown>): NDView<DataType> {
  const [dims, type] = recurseFind(data);
  const flat = Array.isArray(data) ? data.flat(Infinity) : [data];
  // potentially optimizable
  const arr = ndarray(type, dims);
  for (let i = 0; i < flat.length; ++i) {
    arr['t'].b[i] = flat[i];
  }
  return arr;
}

/**
 * Options for `arange`
 */
export interface ArangeOpts<T extends DataType> {
  /**
   * The datatype to use for the output
   */
  dtype?: T;
}

/**
 * Creates a range from 0 to the given stop point, excluding the end
 * @param stop The end of the range
 * @param opts Additional options for range creation
 * @returns A flat ndarray representing the supplied range
 */
export function arange<N extends number, T extends NumericType | BigNumericType = DataType.Int32>(stop: N, opts?: ArangeOpts<T>): NDView<T, [N]>;
/**
* Creates a range from the given start point to the given stop point, excluding the end
* @param start The start of the range
* @param stop The end of the range
* @param opts Additional options for range creation
* @returns A flat ndarray representing the supplied range
*/
export function arange<T extends NumericType | BigNumericType = DataType.Float64 | DataType.Int32>(start: number, stop: number, opts?: ArangeOpts<T>): NDView<T, [number]>;
/**
 * Creates a stepped range from the given start point to the given stop point, excluding the end
 * @param start The start of the range
 * @param stop The end of the range
 * @param step The number to increment by between each element
 * @param opts Additional options for range creation
 * @returns A flat ndarray representing the supplied range
 */
export function arange<T extends NumericType | BigNumericType = DataType.Float64 | DataType.Int32>(start: number, stop: number, step: number, opts?: ArangeOpts<T>): NDView<T, [number]>;
export function arange<T extends NumericType | BigNumericType>(stopOrStart?: number, startOrStopOrOpts?: number | ArangeOpts<T>, stepOrOpts?: number | ArangeOpts<T>, opts?: ArangeOpts<T>) {
  let start: number, stop: number, step: number, dtype: T
  if (typeof opts == 'object' || typeof stepOrOpts == 'number') {
    start = stopOrStart;
    stop = startOrStopOrOpts as number;
    step = stepOrOpts as number;
    dtype = opts && opts.dtype;
  } else if (typeof stepOrOpts == 'object' || typeof startOrStopOrOpts == 'number') {
    start = stopOrStart;
    stop = startOrStopOrOpts as number;
    step = 1;
    dtype = stepOrOpts && (stepOrOpts as ArangeOpts<T>).dtype;
  } else {
    start = 0;
    stop = stopOrStart;
    step = 1;
    dtype = startOrStopOrOpts && (startOrStopOrOpts as ArangeOpts<T>).dtype;
  }
  if (!dtype) {
    dtype = (Number.isInteger(start) && Number.isInteger(stop) && Number.isInteger(step) ? DataType.Int32 : DataType.Float64) as T;
  }
  const len = Math.max(Math.floor((stop - start) / step), 0);
  const arr = ndarray(dtype, [len] as [number]);
  const view = arr[ndvInternals];
  if ((dtype as DataType) == DataType.Int64 || (dtype as DataType) == DataType.Uint64) {
    for (let i = start, ind = 0; ind < len; i += step, ++ind) {
      view['t'].b[ind] = BigInt(i);
    }
  } else {
    for (let i = start, ind = 0; ind < len; i += step, ++ind) view['t'].b[ind] = i;
  }
  return arr;
}

/**
 * Options for `zeros`
 */
export interface ZerosOpts<T extends DataType> {
  /**
   * The datatype to use for the output
   */
  dtype?: T;
}

/**
 * Creates an array of zeros
 * @param shape The shape of the ndarray
 * @param opts Additional options for array creation
 * @returns A zeroed ndarray of the given shape
 */
export function zeros<D extends Dims, T extends NumericType | BigNumericType = DataType.Int32>(shape: D, opts?: ZerosOpts<T>): NDView<T, D> {
  const arr = ndarray((opts && opts.dtype || DataType.Int32) as T, shape);
  return arr;
}

/**
 * Options for `ones`
 */
export interface OnesOpts<T extends DataType> {
  /**
   * The datatype to use for the output
   */
  dtype?: T;
}

/**
 * Creates an array of ones
 * @param shape The shape of the ndarray
 * @param opts Additional options for array creation
 * @returns A ndarray full of ones with the given shape
 */
export function ones<D extends Dims, T extends NumericType | BigNumericType = DataType.Int32>(shape: D, opts?: OnesOpts<T>): NDView<T, D> {
  const arr = ndarray((opts && opts.dtype || DataType.Int32) as T, shape);
  arr.set(
    arr.dtype == DataType.Int64 || arr.dtype == DataType.Uint64 ? BigInt(1) : 1
  );
  return arr;
}