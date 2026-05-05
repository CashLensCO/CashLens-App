import { useMemo } from 'react';
import { runAtTargetFps, useFrameProcessor } from 'react-native-vision-camera';
import { useRunOnJS, useSharedValue } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import type { TensorflowModel } from 'react-native-fast-tflite';
import { SHAPE_LABELS, BILL_LABELS } from '../utils/labels';
import { smoothDetection, DetectionResult } from '../utils/smoothing';

const INFERENCE_FPS = 3;
const COIN_SHAPE_THRESHOLD = 0.93;
const BILL_SHAPE_THRESHOLD = 0.3;
// Por debajo de este umbral consideramos la denominación "incierta" y
// emitimos un estado de retry para pedirle al usuario reintentar.
const RETRY_DENOM_THRESHOLD = 0.15;
const OOM_BACKOFF_MS = 1200;
const ERROR_LOG_INTERVAL_MS = 2000;
const SHAPE_BILL_INDEX = SHAPE_LABELS.indexOf('bill');
const SHAPE_COIN_INDEX = SHAPE_LABELS.indexOf('coin');

type ResizeDataType = 'uint8' | 'float32';

type ModelInputConfig = {
  width: number;
  height: number;
  dataType: ResizeDataType;
};

function getModelInputConfig(model: TensorflowModel | undefined): ModelInputConfig | null {
  if (!model || model.inputs.length === 0) return null;

  const input = model.inputs[0];
  const shape = input.shape;

  const asPositive = (value: number | undefined): number | undefined => {
    if (typeof value !== 'number') return undefined;
    if (!Number.isFinite(value) || value <= 0) return undefined;
    return Math.round(value);
  };

  let height: number | undefined;
  let width: number | undefined;

  if (shape.length >= 4) {
    const s1 = asPositive(shape[1]);
    const s2 = asPositive(shape[2]);
    const s3 = asPositive(shape[3]);

    const channelsLast = s3 === 1 || s3 === 3;
    const channelsFirst = s1 === 1 || s1 === 3;

    if (channelsLast) {
      height = s1;
      width = s2;
    } else if (channelsFirst) {
      height = s2;
      width = s3;
    } else {
      height = s1;
      width = s2;
    }
  } else if (shape.length === 3) {
    const s0 = asPositive(shape[0]);
    const s1 = asPositive(shape[1]);
    const s2 = asPositive(shape[2]);

    const channelsLast = s2 === 1 || s2 === 3;
    const channelsFirst = s0 === 1 || s0 === 3;

    if (channelsLast) {
      height = s0;
      width = s1;
    } else if (channelsFirst) {
      height = s1;
      width = s2;
    } else {
      height = s0;
      width = s1;
    }
  }

  if (!height || !width) return null;

  const dataType: ResizeDataType = input.dataType === 'float32' ? 'float32' : 'uint8';
  return { width, height, dataType };
}

function softmax(arr: ArrayLike<number>): number[] {
  'worklet';
  let max = arr[0] ?? 0;
  for (let i = 1; i < arr.length; i++) {
    const value = arr[i] ?? 0;
    if (value > max) max = value;
  }

  const exps: number[] = [];
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const value = Math.exp((arr[i] ?? 0) - max);
    exps.push(value);
    sum += value;
  }

  if (sum <= 0) return exps.map(() => 0);
  return exps.map(x => x / sum);
}

function toProbabilities(arr: ArrayLike<number>): number[] {
  'worklet';
  const values: number[] = [];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sum = 0;

  for (let i = 0; i < arr.length; i++) {
    const value = Number.isFinite(arr[i] ?? NaN) ? (arr[i] as number) : 0;
    values.push(value);
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
  }

  const looksLikeProbabilities =
    values.length > 0 &&
    min >= -0.001 &&
    max <= 1.001 &&
    sum >= 0.98 &&
    sum <= 1.02;

  return looksLikeProbabilities ? values : softmax(values);
}

function argmax(arr: ArrayLike<number>): number {
  'worklet';
  let maxIdx = 0;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? 0) > (arr[maxIdx] ?? 0)) maxIdx = i;
  }
  return maxIdx;
}

export function useCameraFrameProcessor(
  shapeModel: TensorflowModel | undefined,
  // coinsModel ya no se usa: OCR (ML Kit) reemplaza al TFLite para monedas.
  // Se mantiene en la firma para no romper el llamado desde CameraScreen.
  _coinsModel: TensorflowModel | undefined,
  billsModel: TensorflowModel | undefined,
  onResult: (result: DetectionResult) => void,
  onFps: (fps: number) => void,
) {
  const { resize } = useResizePlugin();

  const shapeInputConfig = useMemo(() => getModelInputConfig(shapeModel), [shapeModel]);
  const billsInputConfig = useMemo(() => getModelInputConfig(billsModel), [billsModel]);

  const processedCount = useSharedValue(0);
  const fpsTimer = useSharedValue(0);
  const lastErrorLogTime = useSharedValue(0);
  const oomCooldownUntil = useSharedValue(0);

  const runOnFps = useRunOnJS(onFps, [onFps]);

  const runOnResult = useRunOnJS((result: DetectionResult) => {
    const smoothed = smoothDetection(result);
    onResult(smoothed);
  }, [onResult]);

  const runOnLog = useRunOnJS((message: string) => {
    console.log(message);
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';

    if (!shapeModel || !billsModel) return;
    if (!shapeInputConfig || !billsInputConfig) return;

    const now = Date.now();
    if (now < oomCooldownUntil.value) return;

    // Hard-cap inference rate to lower native buffer allocations.
    runAtTargetFps(INFERENCE_FPS, () => {
      'worklet';

      const inferNow = Date.now();
      if (inferNow < oomCooldownUntil.value) return;

      processedCount.value += 1;
      if (inferNow - fpsTimer.value >= 1000) {
        const fps = processedCount.value;
        processedCount.value = 0;
        fpsTimer.value = inferNow;
        runOnFps(fps);
      }

      try {
        const shapeInput = resize(frame, {
          scale: {
            width: shapeInputConfig.width,
            height: shapeInputConfig.height,
          },
          pixelFormat: 'rgb',
          dataType: shapeInputConfig.dataType,
        });

        const shapeOutput = shapeModel.runSync([shapeInput]);
        const shapeScores = shapeOutput[0] as Float32Array;
        const shapeProbs = toProbabilities(shapeScores);
        const billShapeConf = shapeProbs[SHAPE_BILL_INDEX] ?? 0;
        const coinShapeConf = shapeProbs[SHAPE_COIN_INDEX] ?? 0;

        const coinEligible = coinShapeConf >= COIN_SHAPE_THRESHOLD;
        const billEligible = billShapeConf >= BILL_SHAPE_THRESHOLD;

        if (!coinEligible && !billEligible) {
          runOnResult({ type: 'none', label: '', confidence: 0 });
          return;
        }

        const targetType: 'coin' | 'bill' =
          coinShapeConf >= billShapeConf ? 'coin' : 'bill';

        if (targetType === 'coin' && coinEligible) {
          // Delegamos la denominación de monedas al pipeline de OCR (ML Kit)
          // en CameraScreen: con shape confiable ya sabemos que hay una
          // moneda en frame; OCR lee el número impreso, que es mucho más
          // robusto que el modelo TFLite para monedas colombianas.
          runOnResult({
            type: 'coin_analyzing',
            label: '',
            confidence: coinShapeConf,
          });
          return;
        }

        if (targetType === 'bill' && billEligible) {
          const billInput =
            billsInputConfig.width === shapeInputConfig.width &&
            billsInputConfig.height === shapeInputConfig.height &&
            billsInputConfig.dataType === shapeInputConfig.dataType
              ? shapeInput
              : resize(frame, {
                  scale: {
                    width: billsInputConfig.width,
                    height: billsInputConfig.height,
                  },
                  pixelFormat: 'rgb',
                  dataType: billsInputConfig.dataType,
                });

          const billOutput = billsModel.runSync([billInput]);
          const billScores = billOutput[0] as Float32Array;
          const billProbs = toProbabilities(billScores);
          const billIdx = argmax(billProbs);
          const billConf = billProbs[billIdx] ?? 0;

          if (billConf < RETRY_DENOM_THRESHOLD) {
            runOnResult({ type: 'retry_bill', label: '', confidence: billConf });
            return;
          }

          runOnResult({
            type: 'bill',
            label: BILL_LABELS[billIdx],
            confidence: billConf,
          });
          return;
        }
      } catch (e) {
        const message = String(e);
        if (message.includes('OutOfMemoryError')) {
          oomCooldownUntil.value = Date.now() + OOM_BACKOFF_MS;
        }

        if (inferNow - lastErrorLogTime.value > ERROR_LOG_INTERVAL_MS) {
          lastErrorLogTime.value = inferNow;
          runOnLog(`[FrameProcessor] inference error: ${message}`);
        }
      }
    });
  }, [
    shapeModel,
    billsModel,
    shapeInputConfig,
    billsInputConfig,
    resize,
    runOnResult,
    runOnFps,
    runOnLog,
    fpsTimer,
    processedCount,
    lastErrorLogTime,
    oomCooldownUntil,
  ]);

  return frameProcessor;
}