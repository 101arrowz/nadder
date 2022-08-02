const { readFileSync, writeFileSync } = require('fs');
const { resolve } = require('path');

const dataTypeDTS = resolve(__dirname, '..', 'lib', 'core', 'datatype.d.ts');
const content = readFileSync(dataTypeDTS, 'utf-8');
writeFileSync(dataTypeDTS, content.replace(/const enum/, 'enum'));