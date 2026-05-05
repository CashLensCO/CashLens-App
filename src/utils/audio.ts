import { Image } from 'react-native';
import Sound from 'react-native-sound';
import { Asset } from 'expo-asset';
import { AUDIO_MAP } from './labels';

Sound.setCategory('Playback');

let currentSound: Sound | null = null;
let lastPlayedKey = '';
let lastPlayedTime = 0;
let lastPositivePlayTime = 0; // última vez que sonó coin_* o bill_*
const MIN_INTERVAL_MS = 2500;
// No interrumpir una detección real con un retry si se acaba de anunciar:
const POSITIVE_COOLDOWN_MS = 2500;

function resolveAudioUri(audioFile: any): string {
  let uri = '';
  const imageResolver = (Image as any).resolveAssetSource;
  if (typeof imageResolver === 'function') {
    uri = imageResolver(audioFile)?.uri ?? '';
  }
  if (!uri) {
    const asset = Asset.fromModule(audioFile);
    uri = asset.localUri ?? asset.uri;
  }
  return uri;
}

function playAudioKey(key: string) {
  const audioFile = AUDIO_MAP[key];
  if (!audioFile) return;

  const uri = resolveAudioUri(audioFile);
  if (!uri) return;

  // Detener audio anterior
  if (currentSound) {
    currentSound.stop();
    currentSound.release();
    currentSound = null;
  }

  const sound = new Sound(uri, '', (error) => {
    if (error) {
      console.log('Error cargando audio:', error);
      return;
    }
    sound.play((success) => {
      if (!success) console.log('Error reproduciendo audio');
      sound.release();
    });
  });

  currentSound = sound;
}

export function playInitAudio() {
  playAudioKey('init');
  lastPlayedKey = 'init';
  lastPlayedTime = Date.now();
}

export function playDetectionAudio(
  type:
    | 'coin'
    | 'bill'
    | 'none'
    | 'retry_coin'
    | 'retry_bill'
    | 'coin_analyzing'
    | 'change_coin',
  label: string,
) {
  const now = Date.now();

  let key = '';
  if (type === 'coin') key = `coin_${label}`;
  else if (type === 'bill') key = `bill_${label}`;
  else if (type === 'retry_coin') key = 'retry_coin';
  else if (type === 'retry_bill') key = 'retry_bill';
  else if (type === 'change_coin') key = 'change_coin';
  else return; // 'none' y 'coin_analyzing' no reproducen nada

  // No repetir el mismo audio si no pasó suficiente tiempo
  if (key === lastPlayedKey && now - lastPlayedTime < MIN_INTERVAL_MS) {
    return;
  }

  // Un retry/change no debe interrumpir una detección real recién anunciada
  const isSoftCue =
    key === 'retry_coin' || key === 'retry_bill' || key === 'change_coin';
  if (isSoftCue && now - lastPositivePlayTime < POSITIVE_COOLDOWN_MS) {
    return;
  }

  playAudioKey(key);

  lastPlayedKey = key;
  lastPlayedTime = now;
  if (!isSoftCue && (type === 'coin' || type === 'bill')) {
    lastPositivePlayTime = now;
  }
}
