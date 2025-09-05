// main.js
const { app, BrowserWindow, ipcMain, systemPreferences } = require('electron');
const path = require('path');
const { runWaveform } = require('./ai/load-node');

const createWindow = () => {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        }
    });

    // --- NEW: Handle microphone permission request from the renderer ---
    ipcMain.handle('microphone:get-permission', async () => {
        const granted = await systemPreferences.askForMediaAccess('microphone');
        console.log(`Microphone access granted: ${granted}`);
        return granted;
    });

    // Handle the 'onnx:run' event from the renderer process.
    ipcMain.handle('onnx:run', async (event, arrayBuffer) => {
        console.log('Main process received "onnx:run" event.');
        try {
                // arrayBufferLike should be an ArrayBuffer containing Float32 samples at 32 kHz mono
                const float32 = new Float32Array(arrayBufferLike); // reconstruct typed array
                const tensor = await runWaveform(float32);
                // Return a plain array to the renderer
                return { success: true, embedding: Array.from(tensor.data) };
        } catch (error) {
            console.error('Failed to run ONNX model:', error);
            return { success: false, error: error.message };
        }
    });

    win.loadFile(path.join(__dirname, 'index.html'));
};

app.whenReady().then(() => {
    createWindow();
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

