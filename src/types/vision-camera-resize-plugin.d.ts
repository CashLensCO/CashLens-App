declare module 'vision-camera-resize-plugin' {
  import type { Frame } from 'react-native-vision-camera';

  export type DataType = 'uint8' | 'float32';
  export type PixelFormat = 'rgb' | 'rgba' | 'argb' | 'bgra' | 'bgr' | 'abgr';

  export interface ResizeSize {
    width: number;
    height: number;
  }

  export interface ResizeRect extends ResizeSize {
    x: number;
    y: number;
  }

  export interface ResizeOptions<T extends DataType = DataType> {
    mirror?: boolean;
    crop?: ResizeRect;
    scale?: ResizeSize;
    rotation?: '0deg' | '90deg' | '180deg' | '270deg';
    pixelFormat: PixelFormat;
    dataType: T;
  }

  export interface ResizePlugin {
    resize<T extends DataType>(
      frame: Frame,
      options: ResizeOptions<T>,
    ): T extends 'uint8' ? Uint8Array : Float32Array;
  }

  export function createResizePlugin(): ResizePlugin;
  export function useResizePlugin(): ResizePlugin;
}
