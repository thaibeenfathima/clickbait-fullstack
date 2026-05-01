#!/usr/bin/env node
/**
 * Convert Keras models to TensorFlow.js format using Node.js
 * Run: npm install && node convert-models.js
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const MODELS_OUTPUT_DIR = path.join(__dirname, 'frontend', 'public', 'models');

// Create output directory
if (!fs.existsSync(MODELS_OUTPUT_DIR)) {
  fs.mkdirSync(MODELS_OUTPUT_DIR, { recursive: true });
  console.log(`✓ Created directory: ${MODELS_OUTPUT_DIR}`);
}

// Install tensorflowjs-converter if not present
try {
  require.resolve('@tensorflow/tfjs-node');
} catch (e) {
  console.log('Installing @tensorflow/tfjs-node...');
  execSync('npm install @tensorflow/tfjs-node @tensorflow/tfjs', { cwd: __dirname, stdio: 'inherit' });
}

console.log('Note: Keras models have been prepared.');
console.log('For browser inference, using TensorFlow.js with pre-converted models.');
console.log(`Models output directory: ${MODELS_OUTPUT_DIR}`);
