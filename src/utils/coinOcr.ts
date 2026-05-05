import TextRecognition from '@react-native-ml-kit/text-recognition';
import { COIN_LABELS } from './labels';

// Ordenadas de mayor a menor longitud para evitar que "100" matchee dentro de "1000".
const COIN_DENOMS = [...COIN_LABELS].sort((a, b) => b.length - a.length);

export type CoinOcrResult =
  // OCR reconoció dígitos que coinciden con una denominación.
  | { kind: 'match'; label: string }
  // OCR reconoció texto/dígitos pero ninguno matchea con una denominación.
  | { kind: 'no_match' }
  // OCR no vio ningún dígito en la imagen: probablemente el ciego está
  // mostrando el lado sin número de la moneda.
  | { kind: 'no_text' };

/**
 * Ejecuta OCR sobre una foto de moneda y devuelve:
 *  - match: denominación encontrada.
 *  - no_match: se leyeron dígitos pero ninguno coincide.
 *  - no_text: no se leyó ningún dígito (cara no numérica).
 */
export async function identifyCoinByOcr(imageUri: string): Promise<CoinOcrResult> {
  try {
    const result = await TextRecognition.recognize(imageUri);
    const text = result?.text ?? '';

    // Grupos de dígitos tal cual aparecen en el texto.
    const digitGroups = text.match(/\d+/g) ?? [];

    // Grupos "extendidos" que pueden incluir separadores de miles (".", ",")
    // y que luego normalizamos a solo dígitos. Ej: "1.000" -> "1000".
    const normalizedGroups = (text.match(/\d[\d.,]*\d/g) ?? []).map(s =>
      s.replace(/[.,]/g, ''),
    );

    const candidates = new Set<string>([...digitGroups, ...normalizedGroups]);

    // Si el OCR no vio ni un solo dígito, muy probablemente es el lado
    // con imagen de la moneda (escudo / figura). Pedimos cambiar la cara.
    if (candidates.size === 0) {
      return { kind: 'no_text' };
    }

    // Match por longitud descendente: "1000" gana sobre "100".
    for (const denom of COIN_DENOMS) {
      if (candidates.has(denom)) return { kind: 'match', label: denom };
    }
    // En la moneda de $1000 el "1" a veces no lo lee el OCR (queda pegado al
    // borde / brillo), y solo vemos "000". Tratamos ese caso como 1000.
    if (candidates.has('000')) return { kind: 'match', label: '1000' };
    return { kind: 'no_match' };
  } catch (e) {
    console.log('[coinOcr] error:', e);
    return { kind: 'no_match' };
  }
}
