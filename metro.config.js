const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver.assetExts.push('tflite');
config.resolver.assetExts.push('txt');
config.resolver.assetExts.push('mp3');
config.resolver.assetExts.push('m4a');

module.exports = config;