// preload.js
const { contextBridge, ipcRenderer } = require('electron');

// Expose a secure API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  runOnnxModel: (arrayBuffer) => ipcRenderer.invoke('onnx:run', arrayBuffer),
  // --- NEW: Expose the permission function to the renderer ---
  getMicrophonePermission: () => ipcRenderer.invoke('microphone:get-permission')
});

