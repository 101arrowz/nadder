export interface GlobalOptions {
  /**
   * Whether or not to prefer storing ndarrays in WebAssembly memory. Defaults
   * to `true`.
   */
  preferWASM: boolean;
  /**
   * Whether or not WASM memory should be freed after each operation. Defaults
   * to `false` in environments where garbage collection can be detected, and
   * `true` elsewhere.
   */
  freeWASM: boolean;
}

/**
 * Global options for the library
 */
export const globalOptions: GlobalOptions = {
  preferWASM: true,
  freeWASM: typeof WeakRef == 'undefined'
};