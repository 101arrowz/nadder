import { DataType } from '../core/datatype';

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

let heap: NDV[] = [];

const resizeHandlers = new Map<number, () => void>();

export function watchResize(key: number, val: () => void) {
  resizeHandlers.set(key, val);
}

export function unwatch(key: number) {
  resizeHandlers.delete(key);
}

export function malloc(size: number, align: number) {
  let oldbuf = wasmExports.memory.buffer;
  const ptr = wasmExports.malloc(size, align);
  if (wasmExports.memory.buffer != oldbuf) {
    for (const val of resizeHandlers.values()) val();
  }
  return ptr;
}

export function free(ptr: number, size: number, align: number) {
  let oldbuf = wasmExports.memory.buffer;
  wasmExports.free(ptr, size, align);
  // should never happen since wasm cant yet shrink memory
  if (wasmExports.memory.buffer != oldbuf) {
    for (const val of resizeHandlers.values()) val();
  }
  return ptr;
}

export function allocHeap(obj: NDV) {
  return heap.push(obj);
}

export function freeHeap(idx: number) {
  heap[idx] = null;
}

export function getHeap(idx: number) {
  return heap[idx - 1];
}

export function loadWASM(src: Uint8Array) {
  const module = new WebAssembly.Module(src);
  const instance = new WebAssembly.Instance(module, {
    env: {
      dtype(id: number) {
        return getHeap(id).t;
      },
      ndim(id: number) {
        return getHeap(id).d.length;
      },
      dim(id: number, ind: number) {
        return getHeap(id).d[ind];
      },
      stride(id: number, ind: number) {
        return getHeap(id).s[ind];
      },
      buf(id: number) {
        return getHeap(id).b;
      },
      buflen(id: number) {
        return getHeap(id).l;
      },
      bufoff(id: number) {
        return getHeap(id).i;
      },
      off(id: number) {
        return getHeap(id).o;
      },
      register(
        dtype: DataType,
        ndim: number,
        daddr: number,
        saddr: number,
        buflen: number,
        baddr: number,
        offset: number,
        boff: number
      ) {
        const dims = new Uint32Array(wasmExports.memory.buffer, daddr, ndim);
        const strides = new Uint32Array(wasmExports.memory.buffer, saddr, ndim);
        return allocHeap({
          t: dtype,
          d: [...dims],
          s: [...strides],
          l: buflen,
          b: baddr,
          o: offset,
          i: boff
        });
      }
    }
  });
  wasmExports = instance.exports;
}