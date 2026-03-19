import React, { useEffect, useRef, useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
  TouchableOpacity,
  Platform,
} from 'react-native';
import {
  Camera,
  useCameraFormat,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';
import { useModels } from '../hooks/useModels';
import { useCameraFrameProcessor } from '../hooks/useFrameProcessor';
import { DetectionResult } from '../utils/smoothing';
import { playDetectionAudio } from '../utils/audio';

export function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const format = useCameraFormat(device, [
    { videoResolution: { width: 480, height: 360 } },
    { fps: 10 },
  ]);
  const cameraFps = format
    ? Math.max(format.minFps, Math.min(10, format.maxFps))
    : undefined;
  const { shapeModel, coinsModel, billsModel, isReady, hasError } = useModels();

  const [result, setResult] = useState<DetectionResult>({
    type: 'none',
    label: '',
    confidence: 0,
  });
  const [fps, setFps] = useState(0);
  const lastAudioResult = useRef('');

  // Solicitar permiso de cámara al montar
  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission]);

  // Reproducir audio cuando cambia el resultado
  useEffect(() => {
    const key = `${result.type}:${result.label}`;
    if (key !== lastAudioResult.current) {
      lastAudioResult.current = key;
      playDetectionAudio(result.type, result.label);
    }
  }, [result]);

  const frameProcessor = useCameraFrameProcessor(
    isReady ? shapeModel : undefined,
    isReady ? coinsModel : undefined,
    isReady ? billsModel : undefined,
    setResult,
    setFps,
  );

  // Pantallas de estado
  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>Se necesita permiso de cámara</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Dar permiso</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>No se encontró cámara</Text>
      </View>
    );
  }

  if (hasError) {
    return (
      <View style={styles.centered}>
        <Text style={styles.message}>Error cargando modelos</Text>
      </View>
    );
  }

  const getResultText = () => {
    if (!isReady) return 'Cargando modelos...';
    if (result.type === 'none') return 'No se detecta dinero';
    if (result.type === 'coin') return `Moneda de $${result.label}`;
    if (result.type === 'bill') return `Billete de $${result.label}`;
    return '';
  };

  const getResultColor = () => {
    if (result.type === 'none') return '#ffffff';
    if (result.type === 'coin') return '#FFD700';
    if (result.type === 'bill') return '#00FF88';
    return '#ffffff';
  };

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        fps={cameraFps}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv"
      />

      {/* HUD superior — FPS y estado del modelo */}
      <View style={styles.hudTop}>
        <Text style={styles.hudText}>
          {isReady ? `${fps} fps` : 'Cargando...'}
        </Text>
        {!isReady && (
          <ActivityIndicator color="#ffffff" size="small" style={{ marginLeft: 8 }} />
        )}
      </View>

      {/* Resultado principal */}
      <View style={[styles.resultContainer, result.type !== 'none' && styles.resultContainerActive]}>
        <Text style={[styles.resultText, { color: getResultColor() }]}>
          {getResultText()}
        </Text>
        {result.type !== 'none' && (
          <>
            <View style={[styles.confidenceBar, { width: `${result.confidence * 100}%`, backgroundColor: getResultColor() }]} />
            <Text style={styles.confidenceText}>
              {Math.round(result.confidence * 100)}% confianza
            </Text>
          </>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 20,
  },
  message: {
    color: '#fff',
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  hudTop: {
    position: 'absolute',
    top: 50,
    left: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  hudText: {
    color: '#ffffff',
    fontSize: 13,
    fontFamily: Platform.OS === 'android' ? 'monospace' : 'Courier New',
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  resultContainer: {
    position: 'absolute',
    bottom: 80,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  resultContainerActive: {
    backgroundColor: 'rgba(0,0,0,0.85)',
    borderColor: 'rgba(255,255,255,0.3)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.8,
    shadowRadius: 12,
    elevation: 10,
  },
  resultText: {
    fontSize: 32,
    fontWeight: '800',
    textAlign: 'center',
    letterSpacing: 0.5,
  },
  confidenceBar: {
    height: 4,
    marginTop: 12,
    marginBottom: 8,
    borderRadius: 2,
    maxWidth: '100%',
  },
  confidenceText: {
    color: 'rgba(255,255,255,0.8)',
    fontSize: 12,
    marginTop: 4,
    fontWeight: '600',
    letterSpacing: 0.3,
  },
});