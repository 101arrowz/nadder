import { DataType, DataTypeBuffer, dataTypeBufferMap } from './datatype';

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  if (Array.isArray(data)) {
    if (data.length > 0 && data.every(d => typeof d == 'string')) return DataType.String as T;
    return DataType.Object as T;
  }
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

export class FlatArray<T extends DataType> {
  // type
  t: T;
  // buffer
  b: DataTypeBuffer<T>;

  constructor(data: DataTypeBuffer<T>);
  constructor(type: T, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size?: number) {
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](size) as DataTypeBuffer<T>;
      if (this.t == DataType.Object) (this.b as unknown[]).fill(null);
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
    }
  }
}