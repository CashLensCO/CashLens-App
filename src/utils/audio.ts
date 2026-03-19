import { Image } from 'react-native';
import Sound from 'react-native-sound';
import { AUDIO_MAP } from './labels';

Sound.setCategory('Playback');

let currentSound: Sound | null = null;
let lastPlayedKey = '';
let lastPlayedTime = 0;
const MIN_INTERVAL_MS = 2500;

export function playDetectionAudio(type: 'coin' | 'bill' | 'none', label: string) {
  const now = Date.now();
  
  let key = '';
  if (type === 'coin') key = `coin_${label}`;
  else if (type === 'bill') key = `bill_${label}`;
  else key = 'no_detection';

  // No repetir el mismo audio si no pasó suficiente tiempo
  if (key === lastPlayedKey && now - lastPlayedTime < MIN_INTERVAL_MS) {
    return;
  }

  const audioFile = AUDIO_MAP[key];
  if (!audioFile) return;

  const { uri } = Image.resolveAssetSource(audioFile);

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
  lastPlayedKey = key;
  lastPlayedTime = now;
}