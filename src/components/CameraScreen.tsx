import React, { useCallback, useEffect, useRef, useState } from 'react';
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
import { playDetectionAudio, playInitAudio } from '../utils/audio';
import { identifyCoinByOcr } from '../utils/coinOcr';

// Lock tras una identificación exitosa: mantiene el resultado visible
// y evita que el frame processor lo pise con coin_analyzing mientras el
// usuario sostiene la moneda.
const COIN_LOCK_MS = 3000;
// Mínimo entre capturas de OCR para no saturar la cámara / ML Kit.
const OCR_MIN_INTERVAL_MS = 900;
// Antes de pedir "voltea la moneda" damos N intentos de retry_coin:
// puede que la moneda esté en la cara numérica pero el OCR todavía no vea
// los dígitos por enfoque / ángulo.
const CHANGE_COIN_AFTER_NO_TEXT = 6;

export function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const format = useCameraFormat(device, [
    { videoResolution: { width: 480, height: 360 } },
    // Para OCR necesitamos una foto nítida del texto de la moneda.
    { photoResolution: { width: 1280, height: 720 } },
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
  const cameraRef = useRef<Camera>(null);

  // Lock del resultado de OCR: {label, until} mientras esté vigente,
  // ignoramos nuevos coin_analyzing y mantenemos la detección visible.
  const lockedCoinRef = useRef<{ label: string; until: number } | null>(null);
  // OCR en vuelo y último tiempo de captura para throttling.
  const ocrInFlight = useRef(false);
  const lastOcrTime = useRef(0);
  // Cuántas veces seguidas el OCR no vio ningún dígito: acumulamos antes
  // de rendirnos y pedir "voltea la moneda".
  const noTextStreakRef = useRef(0);

  // Interponemos este handler entre el frame processor y setResult:
  // - si hay una moneda bloqueada y llega coin_analyzing/none, sostenemos el lock.
  // - si llega cualquier otra cosa (bill, retry_bill) liberamos el lock.
  const handleFrameResult = useCallback((next: DetectionResult) => {
    const now = Date.now();
    const locked = lockedCoinRef.current;

    if (locked && now < locked.until) {
      if (
        next.type === 'coin_analyzing' ||
        next.type === 'none' ||
        (next.type === 'coin' && next.label === locked.label)
      ) {
        setResult({ type: 'coin', label: locked.label, confidence: 1 });
        return;
      }
      // El usuario cambió a billete u otra moneda: liberar el lock.
      lockedCoinRef.current = null;
    } else if (locked) {
      lockedCoinRef.current = null;
    }
    setResult(next);
  }, []);

  // Solicitar permiso de cámara al montar
  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission]);

  // Audio de bienvenida: solo una vez al entrar a la app
  useEffect(() => {
    playInitAudio();
  }, []);

  // Reproducir audio cuando cambia el resultado
  useEffect(() => {
    const key = `${result.type}:${result.label}`;
    if (key !== lastAudioResult.current) {
      lastAudioResult.current = key;
      playDetectionAudio(result.type, result.label);
    }
  }, [result]);

  // Pipeline OCR para monedas: se dispara cuando el frame processor
  // reporta coin_analyzing (forma = moneda con alta confianza).
  useEffect(() => {
    if (result.type !== 'coin_analyzing') return;
    const now = Date.now();
    if (ocrInFlight.current) return;
    if (now - lastOcrTime.current < OCR_MIN_INTERVAL_MS) return;
    const locked = lockedCoinRef.current;
    if (locked && now < locked.until) return;

    ocrInFlight.current = true;
    lastOcrTime.current = now;

    (async () => {
      try {
        const camera = cameraRef.current;
        if (!camera) return;
        const photo = await camera.takePhoto({
          enableShutterSound: false,
          flash: 'off',
        });
        const uri = photo.path.startsWith('file://')
          ? photo.path
          : `file://${photo.path}`;
        const ocr = await identifyCoinByOcr(uri);
        if (ocr.kind === 'match') {
          noTextStreakRef.current = 0;
          lockedCoinRef.current = {
            label: ocr.label,
            until: Date.now() + COIN_LOCK_MS,
          };
          setResult({ type: 'coin', label: ocr.label, confidence: 1 });
        } else if (ocr.kind === 'no_text') {
          // OCR no leyó ningún dígito. Damos varios intentos de retry antes
          // de asumir que están mostrando el lado sin número.
          noTextStreakRef.current += 1;
          if (noTextStreakRef.current > CHANGE_COIN_AFTER_NO_TEXT) {
            setResult({ type: 'change_coin', label: '', confidence: 0 });
          } else {
            setResult({ type: 'retry_coin', label: '', confidence: 0 });
          }
        } else {
          // Se leyeron dígitos pero ninguno matchea una denominación.
          // Reseteamos el streak: están en la cara numérica.
          noTextStreakRef.current = 0;
          setResult({ type: 'retry_coin', label: '', confidence: 0 });
        }
      } catch (e) {
        console.log('[CameraScreen] takePhoto/OCR error:', e);
      } finally {
        ocrInFlight.current = false;
      }
    })();
  }, [result]);

  const frameProcessor = useCameraFrameProcessor(
    isReady ? shapeModel : undefined,
    isReady ? coinsModel : undefined,
    isReady ? billsModel : undefined,
    handleFrameResult,
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
    if (result.type === 'coin_analyzing') return 'Analizando moneda...';
    if (result.type === 'retry_coin') return 'Moneda detectada, intenta de nuevo';
    if (result.type === 'retry_bill') return 'Billete detectado, intenta de nuevo';
    if (result.type === 'change_coin') return 'Voltea la moneda al lado con número';
    return '';
  };

  const getResultColor = () => {
    if (result.type === 'coin') return '#FFD700';
    if (result.type === 'bill') return '#00FF88';
    if (result.type === 'coin_analyzing') return '#66CCFF';
    if (result.type === 'retry_coin' || result.type === 'retry_bill') return '#FFA500';
    if (result.type === 'change_coin') return '#FFA500';
    return '#ffffff';
  };

  const isActiveResult =
    result.type === 'coin' ||
    result.type === 'bill' ||
    result.type === 'coin_analyzing' ||
    result.type === 'retry_coin' ||
    result.type === 'retry_bill' ||
    result.type === 'change_coin';
  const showConfidenceBar = result.type === 'coin' || result.type === 'bill';

  return (
    <View style={styles.container}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        format={format}
        fps={cameraFps}
        isActive={true}
        photo={true}
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
      <View
        style={[styles.resultContainer, isActiveResult && styles.resultContainerActive]}
        accessible
        accessibilityLiveRegion="polite"
        accessibilityLabel={getResultText()}
      >
        <Text style={[styles.resultText, { color: getResultColor() }]}>
          {getResultText()}
        </Text>
        {showConfidenceBar && (
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