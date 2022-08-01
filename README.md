# nadder
Easy n-dimensional data manipulation with NumPy syntax.

## Usage
```js
import { ndarray, Bitset } from 'nadder';

const dataSource = new Float32Array(1_000_000);
// load data into dataSource here...

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

const boolIndex = ndarray(
    Bitset.fromArray(leapYears),
    [1000]
);

// You can even use other ndarrays in the indices!

console.log(t[`..., ${boolIndex}`])
```

## Features
- Ergonomic NumPy slicing
- Performant view-based manipulation; minimal copying
- Fast bitset-backed boolean ndarrays
- Arithmetic operators
- In progress: support for most NumPy manipulations, more fast paths for higher performance, opt-in WASM modules
- TBD: better TypeScript support (difficult due to proxies)

### Limitations
- No direct Fortran (column-major) memory layout support


## License
MIT