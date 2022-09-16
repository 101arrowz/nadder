import { NDView } from '../core/ndarray';
import { DataType, dataTypeBufferMap } from '../core/datatype';
import { aligns, FlatArray } from '../core/flatarray';
import { Bitset, globalOptions } from '../util';

export let wasmExports = null;

type NDV = {
  // type
  t: DataType;
  // strides
  s: number[];
  // dims
  d: number[];
  // offset
  o: number;
  // address
  b: number;
  // length
  l: number;
  // buffer offset (for bitset)
  i: number;
};

const shared: NDV[] = [];

type FreeInfo = {
  // value
  v: { deref(): FlatArray<DataType> | undefined };
  // ptr
  p: number;
  // align
  a: number;
  // size
  s: number;
  // refcount
  r: number;
}

const allocs = new Map<number, FreeInfo>();

export function tagAlloc(val: FlatArray<DataType>, len: number) {
  let ref = typeof WeakRef == 'undefined'
    ? { v: val, deref() { return this.v } }
    : new WeakRef(val);

  allocs.set(val.l, {
    v: ref,
    p: val.l,
    a: aligns[val.t],
    s: len,
    r: 0
  });
}

export function releaseAlloc(ptr: number) {
  const entry = allocs.get(ptr);
  if (entry) {
    free(entry.p, entry.s, entry.a);
    allocs.delete(ptr);
  }
}

const gc = () => {
  for (const val of allocs.values()) {
    if (!val.v.deref()) {
      free(val.p, val.s, val.a);
      allocs.delete(val.p);
    }
  }
}

const reattachBuf = (info: FreeInfo) => {
  const obj = info.v.deref();
  if (obj.t == DataType.Bool) {
    obj.b = new Bitset(new Uint8Array(wasmExports.memory.buffer, info.p, info.s), obj.b.length, (obj.b as Bitset).offset);
  } else {
    obj.b = new (dataTypeBufferMap[obj.t] as Uint8ArrayConstructor)(wasmExports.memory.buffer, info.p, info.s / (dataTypeBufferMap[obj.t] as Uint8ArrayConstructor).BYTES_PER_ELEMENT);
  }
}

export function malloc(size: number, align: number) {
  let oldbuf = wasmExports.memory.buffer;
  const ptr = wasmExports.malloc(size, align);
  gc();
  if (wasmExports.memory.buffer != oldbuf) {
    for (const val of allocs.values()) {
      reattachBuf(val);
    }
  }
  return ptr;
}

export function free(ptr: number, size: number, align: number) {
  let oldbuf = wasmExports.memory.buffer;
  wasmExports.free(ptr, size, align);
  gc();
  // should never happen since wasm cant yet shrink memory
  if (wasmExports.memory.buffer != oldbuf) {
    for (const val of allocs.values()) {
      reattachBuf(val);
    }
  }
  return ptr;
}

export function share(v: NDView) {
  const ndv = {
    t: v.dtype,
    s: v['s'].slice(),
    d: v['d'].slice(),
    o: v['o'],
    b: v['t'].w(),
    l: v['t'].b.length,
    i: v.dtype == DataType.Bool ? (v['t'].b as Bitset).offset : 0
  };
  ++allocs.get(ndv.b).r;
  return shared.push(ndv);
}

export function freeShared(idx: number) {
  if (!idx) return;
  let info = allocs.get(getShared(idx).b);
  if (!--info.r && globalOptions.freeWASM) {
    const val = info.v.deref();
    if (val) val.f();
    allocs.delete(info.p);
  }
  shared[idx - 1] = null;
  for (let i = idx - 1; i < shared.length; ++i) {
    if (shared[i]) return;
  }
  shared.length = idx - 2;
}

function getShared(idx: number) {
  return shared[idx - 1];
}

export function loadWASM(src: Uint8Array) {
  const module = new WebAssembly.Module(src);
  const instance = new WebAssembly.Instance(module, {
    env: {
      dtype(id: number) {
        return getShared(id).t;
      },
      ndim(id: number) {
        return getShared(id).d.length;
      },
      dim(id: number, ind: number) {
        return getShared(id).d[ind];
      },
      stride(id: number, ind: number) {
        return getShared(id).s[ind];
      },
      buf(id: number) {
        return getShared(id).b;
      },
      buflen(id: number) {
        return getShared(id).l;
      },
      bufoff(id: number) {
        return getShared(id).i;
      },
      off(id: number) {
        return getShared(id).o;
      }
    }
  });
  wasmExports = instance.exports;
}