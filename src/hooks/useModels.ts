import { useEffect } from 'react';
import { useTensorflowModel } from 'react-native-fast-tflite';

export function useModels() {
  const shapeModel = useTensorflowModel(
    require('../../assets/ml/shape_detector.tflite')
  );
  const coinsModel = useTensorflowModel(
    require('../../assets/ml/coins_v1.tflite')
  );
  const billsModel = useTensorflowModel(
    require('../../assets/ml/bills_v1.tflite')
  );

  const isReady =
    shapeModel.state === 'loaded' &&
    coinsModel.state === 'loaded' &&
    billsModel.state === 'loaded';

  const hasError =
    shapeModel.state === 'error' ||
    coinsModel.state === 'error' ||
    billsModel.state === 'error';

  useEffect(() => {
    if (shapeModel.state === 'loaded') {
      console.log('[TFLite] shape model input:', shapeModel.model.inputs[0]);
      console.log('[TFLite] shape model output:', shapeModel.model.outputs[0]);
    }
  }, [shapeModel.state, shapeModel.model]);

  useEffect(() => {
    if (coinsModel.state === 'loaded') {
      console.log('[TFLite] coins model input:', coinsModel.model.inputs[0]);
      console.log('[TFLite] coins model output:', coinsModel.model.outputs[0]);
    }
  }, [coinsModel.state, coinsModel.model]);

  useEffect(() => {
    if (billsModel.state === 'loaded') {
      console.log('[TFLite] bills model input:', billsModel.model.inputs[0]);
      console.log('[TFLite] bills model output:', billsModel.model.outputs[0]);
    }
  }, [billsModel.state, billsModel.model]);

  return {
    shapeModel: shapeModel.model,
    coinsModel: coinsModel.model,
    billsModel: billsModel.model,
    isReady,
    hasError,
  };
}