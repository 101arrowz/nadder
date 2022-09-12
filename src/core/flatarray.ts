import { DataType, DataTypeBuffer, dataTypeBufferMap } from './datatype';

const findType = <T extends DataType>(data: DataTypeBuffer<T>): T => {
  for (const key in dataTypeBufferMap) {
    if (data instanceof dataTypeBufferMap[key]) {
      return +key as T;
    }
  }
}

/** @internal */
export class FlatArray<T extends DataType> {
  // type
  readonly t: T;
  // buffer
  readonly b: DataTypeBuffer<T>;

  constructor(data: DataTypeBuffer<T>);
  constructor(type: T, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size: number);
  constructor(dataOrType: T | DataTypeBuffer<T>, size?: number) {
    if (typeof dataOrType == 'number') {
      this.t = dataOrType;
      this.b = new dataTypeBufferMap[this.t](size) as DataTypeBuffer<T>;
      if (this.t == DataType.Any) (this.b as unknown[]).fill(undefined);
    } else {
      this.t = findType(dataOrType);
      this.b = dataOrType;
    }
  }
}