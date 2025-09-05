// src/ai/load-node.js
const ort = require('onnxruntime-node');
const path = require('path');

let sessionPromise = null;

// Resolve the packaged path (works in dev and after packaging)
function modelPath() {
  // When packaged, Electron apps read files from app.asar; resources outside should be in process.resourcesPath
  // Keep the model in '<project>/models/qvim.onnx' and configure your packager to copy 'models' as extraResources.
  const base = process.resourcesPath ?? process.cwd();
  return path.join(base, 'models', 'qvim.onnx');
}

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelPath(), {
      // You can add providers here; CPU is usually fine for audio embeddings
      executionProviders: ['cpu']
    });
  }
  return sessionPromise;
}

/**
 * Runs the model on a mono Float32Array waveform at the model’s sample rate (e.g., 32 kHz).
 * Input shape: (T,) → fed as (1, T)
 */
async function runWaveform(float32Array) {
  const session = await getSession();
  const input = new ort.Tensor('float32', float32Array, [1, float32Array.length]);
  const outputs = await session.run({ waveform: input });
  // Use the named output if you know it (e.g., 'embedding'); otherwise take the first
  const first = outputs.embedding ?? outputs.output ?? Object.values(outputs)[0];
  return first; // This is an ort.Tensor; first.data is the typed array
}

module.exports = { runWaveform };
