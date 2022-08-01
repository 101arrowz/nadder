import { NDView, DataType } from '../core';

type Broadcast<T extends DataType[]> = { [I in keyof T]: NDView<T[I]> };

export function broadcast<T extends DataType[]>(...views: Broadcast<T>) {
  if (views.length < 2) return views;
  let maxDims = 0;
  const allInfo = views.map(v => {
    if (v['d'].length > maxDims) maxDims = v['d'].length;
    return {
      v,
      d: v['d'].slice().reverse(),
      s: v['s'].slice().reverse()
    }
  });
  for (let i = 0; i < maxDims; ++i) {
    const target = allInfo.find(v => i < v.d.length && v.d[i] != 1);
    for (const info of allInfo) {
      if (i >= info.d.length || info.d[i] == 1) {
        info.d[i] = target.d[i];
        info.s[i] = 0;
      } else if (info.d[i] != target.d[i]) {
        throw new TypeError(`could not broadcast ndarray with dims (${info.v.shape.join(', ')}) to (${target.v.shape.join(', ')})`);
      }
    }
  }
  return allInfo.map(info => new NDView(info.v['t'], info.d.reverse(), info.s.reverse(), info.v['o'])) as unknown as Broadcast<T>;
}