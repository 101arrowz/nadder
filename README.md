# nadder
Easy n-dimensional data manipulation with NumPy syntax.

## Usage
```js
import { tensor } from 'nadder';

const dataSource = new Float32Array(1_000_000);
// load data into dataSource here...

// Initialize (1000, 1000) tensor
const t = tensor(dataSource, [1000, 1000]);

// NumPy slicing fully supported
console.log(t['100:110, 300:305']);

t[':100'] = 1.23;
```

## Features
- Ergonomic NumPy slicing
- In progress: support for most NumPy manipulations, more fast paths for higher performance, opt-in WASM modules
- TBD: better TypeScript support (difficult due to proxies)