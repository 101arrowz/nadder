import { NDView, RecursiveArray, array } from '../core/ndarray';
import { DataType, IndexType } from '../core/datatype';
import { ndvInternals } from './internal';

type Broadcastable<T extends DataType[]> = { [I in keyof T]: NDView<T[I]> | RecursiveArray<IndexType<T[I]>> };
type Broadcast<T extends DataType[]> = { [I in keyof T]: NDView<T[I]> };

export function broadcast<T extends DataType[]>(...views: Broadcastable<T>) {
  if (views.length < 2) return views as Broadcast<T>;
  let maxDims = 0;
  const allInfo = views.map(v => {
    if (!v[ndvInternals]) v = array(v);
    if (v['d'].length > maxDims) maxDims = v['d'].length;
    return {
      v: v as NDView,
      d: v['d'].slice().reverse(),
      s: v['s'].slice().reverse()
    };
  });
  for (let i = 0; i < maxDims; ++i) {
    const target = allInfo.find(v => i < v.d.length && v.d[i] != 1) || allInfo.find(v => i < v.d.length);
    for (const info of allInfo) {
      if (i >= info.d.length || info.d[i] == 1) {
        info.d[i] = target.d[i];
        info.s[i] = 0;
      } else if (info.d[i] != target.d[i]) {
        throw new TypeError(`could not broadcast ndarray with dims (${info.v['d'].join(', ')}) to (${target.v['d'].join(', ')})`);
      }
    }
  }
  return allInfo.map(info => new NDView(info.v['t'], info.d.reverse(), info.s.reverse(), info.v['o'])[ndvInternals]) as unknown as Broadcast<T>;
}