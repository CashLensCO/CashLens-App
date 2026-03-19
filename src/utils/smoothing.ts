// Suavizado temporal anti-parpadeo
// Mantiene una ventana de N resultados y retorna el más frecuente
// solo si supera el umbral de confianza

export interface DetectionResult {
  type: 'coin' | 'bill' | 'none';
  label: string;
  confidence: number;
}

const WINDOW_SIZE = 3;
const CONFIDENCE_THRESHOLD = 0.67;

const window: DetectionResult[] = [];

export function smoothDetection(result: DetectionResult): DetectionResult {
  window.push(result);
  if (window.length > WINDOW_SIZE) {
    window.shift();
  }

  if (window.length < 2) {
    return { type: 'none', label: '', confidence: 0 };
  }

  // Contar frecuencia de cada label
  const counts: Record<string, number> = {};
  for (const r of window) {
    if (r.type === 'none') continue;
    const key = `${r.type}:${r.label}`;
    counts[key] = (counts[key] || 0) + 1;
  }

  // Encontrar el más frecuente
  let bestKey = '';
  let bestCount = 0;
  for (const [key, count] of Object.entries(counts)) {
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }

  if (!bestKey) {
    return { type: 'none', label: '', confidence: 0 };
  }

  const [type, label] = bestKey.split(':');
  const nonNoneCount = window.reduce((acc, r) => (r.type === 'none' ? acc : acc + 1), 0);
  const frequency = nonNoneCount > 0 ? bestCount / nonNoneCount : 0;

  let latestConfidence = 0;
  for (let i = window.length - 1; i >= 0; i--) {
    const current = window[i];
    if (current.type === type && current.label === label) {
      latestConfidence = current.confidence;
      break;
    }
  }

  const isCoin = type === 'coin';
  const allowSingleCoinHit = isCoin && bestCount === 1 && latestConfidence >= 0.8;

  if (!allowSingleCoinHit) {
    const minCount = isCoin ? 2 : 2;
    const minFrequency = isCoin ? 0.5 : CONFIDENCE_THRESHOLD;
    if (nonNoneCount < minCount || bestCount < minCount || frequency < minFrequency) {
      return { type: 'none', label: '', confidence: 0 };
    }
  }

  return {
    type: type as 'coin' | 'bill',
    label,
    confidence: latestConfidence > 0 ? latestConfidence : frequency,
  };
}

export function resetSmoothingWindow() {
  window.length = 0;
}