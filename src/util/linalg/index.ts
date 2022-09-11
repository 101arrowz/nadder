import { array, DataType, ndarray, NDView } from '../../core';
import { dataTypeNames, IndexType } from '../../core/datatype';
import { Broadcastable } from '../broadcast';
import { makeOut, ndvInternals, UnionToIntersection } from '../internal';

type MT = readonly DataType[];

type Calc<I extends MT, O extends DataType> = (a: IndexType<I[number]>, b: IndexType<I[number]>) => IndexType<O>;

type Impl<I extends MT, O extends DataType> = [args: I, output: O, add: Calc<I, O>, mul: Calc<I, O>, zero: () => IndexType<I[number]>];

const impl = <I extends MT, O extends DataType>(
  inputs: I,
  output: O,
  add: Calc<I, O>,
  mul: Calc<I, O>,
  zero: () => IndexType<I[number]>
): Impl<I, O> => [inputs, output, add, mul, zero];

const numAdd = (a: number, b: number) => a + b;
const numMul = (a: number, b: number) => a * b;
const numZero = () => 0;

const bigAdd = (a: number | bigint, b: number | bigint) => {
  if (typeof a != 'bigint') a = BigInt(a);
  if (typeof b != 'bigint') b = BigInt(b);
  return a + b;
}

const bigMul = (a: number | bigint, b: number | bigint) => {
  if (typeof a != 'bigint') a = BigInt(a);
  if (typeof b != 'bigint') b = BigInt(b);
  return a * b;
}

const bigZero = () => BigInt(0);

const numTypes = [
 impl([DataType.Bool] as const, DataType.Bool, (a, b) => a || b, (a, b) => a && b, () => false),
 impl([DataType.Bool, DataType.Uint8] as const, DataType.Uint8, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Uint8, DataType.Uint8Clamped] as const, DataType.Uint8Clamped, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Int8] as const, DataType.Int8, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16] as const, DataType.Uint16, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16] as const, DataType.Int16, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32] as const, DataType.Uint32, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32] as const, DataType.Int32, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Uint8, DataType.Uint8Clamped, DataType.Uint16, DataType.Uint32, DataType.Uint64] as const, DataType.Uint64, bigAdd, bigMul, bigZero),
 impl([DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Int64] as const, DataType.Int64, bigAdd, bigMul, bigZero),
 impl([DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Float32] as const, DataType.Float32, numAdd, numMul, numZero),
 impl([DataType.Bool, DataType.Int8, DataType.Uint8, DataType.Uint8Clamped, DataType.Int16, DataType.Uint16, DataType.Int32, DataType.Uint32, DataType.Float32, DataType.Int64, DataType.Uint64, DataType.Float64] as const, DataType.Float64, numAdd, numMul, numZero),
] as const;

const typeVals: number[] = numTypes.map(im => im[0].reduce((a, b) => a | b, 0));

type NT = typeof numTypes;

type Sig<T> = T extends Impl<infer I, infer O>
  ? <RO extends DataType = O>(a: Broadcastable<I[number]>, b: Broadcastable<I[number]>, args?: Args<I, RO>) => NDView<RO>
  : never;

type Args<I extends MT, O extends DataType> = {
  out?: NDView<O>;
  dtype?: O;
};

export const matmul = ((a: NDView<DataType>, b: NDView<DataType>, args?: Args<MT, DataType>) => {
  a = a && a[ndvInternals] || array(a)[ndvInternals];
  b = b && b[ndvInternals] || array(b)[ndvInternals];
  
  let outType: DataType = 0;
  let add: Calc<MT, DataType>;
  let mul: Calc<MT, DataType>;
  let zero: () => IndexType<DataType>;
  for (let i = 0; i < typeVals.length; ++i) {
    if ((args && args.dtype == numTypes[i][1]) || typeVals[i] & a['t'].t & b['t'].t) {
      outType = numTypes[i][1];
      add = numTypes[i][2];
      mul = numTypes[i][3];
      zero = numTypes[i][4];
      break;
    }
  }

  if (!outType) {
    throw new TypeError(`cannot apply matmul to tensors of types ${dataTypeNames[a['t'].t]} and (${dataTypeNames[b['t'].t]})`);
  }

  let aUp = 0, bUp = 0;

  if (a.ndim == 1) {
    a = new NDView(a['t'], [1, a['d'][0]], [0, a['s'][0]], a['o'])[ndvInternals];
    aUp = 1;
  }
  if (b.ndim == 1) {
    b = new NDView(b['t'], [b['d'][0], 1], [b['s'][0], 0], b['o'])[ndvInternals];
    bUp = 1;
  }

  if (a.ndim < 2 || b.ndim < 2 || a['d'][a.ndim - 1] != b['d'][b.ndim - 2]) {
    throw new TypeError(`cannot apply matmul to tensors of dimensions (${a['d'].join(', ')}) and (${b['d'].join(', ')})`);
  }
  
  if (b.ndim > a.ndim) {
    a = new NDView(a['t'], [...b['d'].slice(0, b.ndim - a.ndim), ...a['d']], [...b['s'].slice(0, b.ndim - a.ndim).map(() => 0), ...a['s']], a['o'])[ndvInternals]
  } else if (a.ndim > b.ndim){
    b = new NDView(b['t'], [...a['d'].slice(0, a.ndim - b.ndim), ...b['d']], [...a['s'].slice(0, a.ndim - b.ndim).map(() => 0), ...b['s']], b['o'])[ndvInternals]
  }

  const shape = [...a['d'].slice(0, -1), b['d'][b.ndim - 1]];
  const ndim = shape.length;
  const rInd = ndim - 2;
  const cInd = ndim - 1;

  let out = makeOut('matmul', shape.slice(aUp, shape.length - bUp), outType, args && args.out);

  if (aUp) {
    out['d'].unshift(1);
    out['s'].unshift(0);
  }
  if (bUp) {
    out['d'].push(1);
    out['s'].push(0);
  }

  const inner = (dim: number, aInd: number, bInd: number, oInd: number) => {
    if (dim == rInd) {
      const r = shape[rInd];
      const c = shape[cInd];
      const n = a['d'][cInd];
      for (let i = 0; i < r; ++i) {
        for (let j = 0; j < c; ++j) {
          const tgt = oInd + j * out['s'][cInd];
          const lbInd = bInd + j * b['s'][cInd];
          let sum = zero();
          for (let k = 0; k < n; ++k) {
            sum = add(sum, mul(
              a['t'].b[aInd + k * a['s'][cInd]],
              b['t'].b[lbInd + k * b['s'][rInd]]
            ));
          }
          out['t'].b[tgt] = sum;
        }
        oInd += out['s'][rInd];
        aInd += a['s'][rInd];
      }
    } else {
      for (let i = 0; i < a['d'][dim]; ++i) {
        inner(dim + 1, aInd, bInd, oInd)
        aInd += a['s'][dim];
        bInd += b['s'][dim];
        oInd += out['s'][dim];
      }
    }
  }
  inner(0, a['o'], b['o'], out['o']);

  if (aUp) {
    out['d'].shift();
    out['s'].shift();
  }
  if (bUp) {
    out['d'].pop();
    out['s'].pop();
  }

  return out.ndim ? out : out['t'].b[out['o']];
}) as UnionToIntersection<Sig<NT[number]>>;


type IdentityArgs<T extends DataType> = { dtype?: T };

export const identity = <N extends number, T extends DataType = DataType.Int32>(n: N, args?: IdentityArgs<T>) => {
  const nd = ndarray((args && args.dtype || DataType.Int32) as T, [n, n] as [N, N]);
  let one = nd.dtype == DataType.Int64 || nd.dtype == DataType.Uint64
    ? BigInt(1)
    : 1;
  for (let i = 0; i < n; ++i) nd[i][i] = one;
  return nd;
}