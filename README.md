# nadder
Easy n-dimensional data manipulation with NumPy syntax.

## Installation
```sh
npm i nadder # or yarn add nadder, or pnpm add nadder
```

## Usage
```js
import { ndarray, array, add, evaluate, arange } from 'nadder';

const dataSource = new Float32Array(1_000_000);
// load data into dataSource manually here...

// Initialize (1000, 1000) ndarray
const t = ndarray(dataSource, [1000, 1000]);

// NumPy slicing fully supported
console.log(t['100:110, 305:300:-1']);

t['..., :100'] = 1.23;

// np.newaxis is now +
console.log(t['0, 0, +']);

const leapYears = Array.from(
    { length: 1000 },
    (_, i) => i % 4 == 0 && (i % 400 == 0 || i % 100 != 0)
);

// nadder.array automatically creates an efficient representation of your
// Array object and translates nested lists into dimensions in the ndarray
const boolIndex = array(leapYears);

// You can even use other ndarrays in the indices!

console.log(t[`..., ${boolIndex}`]);

// You can evaluate things using a Python-esque DSL
console.log(evaluate`${t}[:, 0] * 2 + 1 / ${arange(1000)}`)

// ufuncs also supported through explicit syntax
// broadcasting, typecasting done automatically
console.log(add(t, boolIndex));
```

## Features
- Ergonomic NumPy slicing, broadcasting
  - All NumPy bracket syntax and indexing routines supported
- NumPy syntax and evaluation via `evaluate`
  - Full support for arithmetic, advanced ops, etc.
- Tiny: 20kB minified, 8kB gzipped
- Performant view-based manipulation; minimal copying
- Fast bitset-backed boolean ndarrays
- Interleaved complex numbers
- Arithmetic, algebraic, and trigonometric operations for real and complex numbers
- Full TypeScript support
- In progress: support for most NumPy manipulations, more fast paths for higher performance, fast WASM modules

### Limitations
- No direct Fortran (column-major) memory layout support
- Limited Complex64 support (Complex128 fully supported)

## License
MIT