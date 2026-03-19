import React from 'react';
import { StatusBar, LogBox } from 'react-native';
import { CameraScreen } from './src/components/CameraScreen';

// Ignorar warnings conocidos de dependencias nativas
LogBox.ignoreLogs([
	'new NativeEventEmitter',
	'Module RNSound',
	'Sending `onAnimatedValueUpdate`',
]);

export default function App() {
	return (
		<>
			<StatusBar barStyle="light-content" backgroundColor="#000" />
			<CameraScreen />
		</>
	);
}
