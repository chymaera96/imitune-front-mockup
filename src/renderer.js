// renderer.js
// TODO: Update based on current ONNX setup
//1.  Decode the recorded Blob into PCM with audioContext.decodeAudioData.
//2.  Resample to 32 kHz with an OfflineAudioContext.
//3.  Extract exactly the window you want (e.g., last 10 s), get a Float32Array.
//4. Send the Float32Array’s buffer to runOnnxModel.

const recordBtn = document.getElementById('recordBtn');
const statusDiv = document.getElementById('status');
const resultsDiv = document.getElementById('results');
const queryAudioSection = document.getElementById('query-audio-section');
const waveformDiv = document.getElementById('waveform');
const playBtn = document.getElementById('playBtn');
const resultList = document.getElementById('result-list');
const downloadBtn = document.getElementById('downloadBtn');

// Create a single, persistent AudioContext to handle all audio operations.
const audioContext = new (window.AudioContext || window.webkitAudioContext)();

let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let wavesurfer;
let recordedAudioUrl = null;

// Initialize WaveSurfer
function initializeWaveSurfer() {
    if (wavesurfer) {
        wavesurfer.destroy();
    }
    wavesurfer = WaveSurfer.create({
        container: waveformDiv,
        waveColor: '#475569',
        progressColor: '#0ea5e9',
        cursorColor: '#fde047',
        barWidth: 3,
        barRadius: 3,
        responsive: true,
        height: 80,
        // Pass the globally managed and resumed AudioContext
        audioContext: audioContext,
    });

    wavesurfer.on('ready', () => {
        console.log('Waveform is ready for playback!');
        playBtn.disabled = false;
        playBtn.textContent = '▶';
    });

    wavesurfer.on('play', () => { playBtn.textContent = '❚❚'; });
    wavesurfer.on('pause', () => { playBtn.textContent = '▶'; });
    wavesurfer.on('finish', () => { playBtn.textContent = '▶'; });
    wavesurfer.on('error', (err) => {
        console.error('Wavesurfer error:', err);
        statusDiv.textContent = `Error visualizing audio: ${err}`;
    });
}

playBtn.addEventListener('click', () => {
    // A failsafe check, though the context should already be running.
    if (audioContext.state === 'suspended') {
        audioContext.resume().then(() => {
            if (wavesurfer && wavesurfer.isPlaying()) {
                 wavesurfer.pause();
            } else if (wavesurfer) {
                 wavesurfer.play();
            }
        });
    } else {
        if (wavesurfer) {
            wavesurfer.playPause();
        }
    }
});

recordBtn.addEventListener('click', async () => {
    // --- FINAL FIX: Resume (unlock) the AudioContext on the first user click ---
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }

    if (isRecording) {
        mediaRecorder.stop();
    } else {
        const hasPermission = await window.electronAPI.getMicrophonePermission();
        if (!hasPermission) {
            statusDiv.textContent = 'Error: Microphone access was denied.';
            return;
        }

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                const audioTracks = stream.getAudioTracks();
                if (audioTracks.length === 0 || audioTracks[0].readyState !== 'live') {
                    console.error('No live audio tracks found.');
                    statusDiv.textContent = 'Error: Could not find an active microphone.';
                    return;
                }

                const options = { mimeType: 'audio/webm;codecs=opus' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    delete options.mimeType;
                }

                mediaRecorder = new MediaRecorder(stream, options);
                audioChunks = [];

                mediaRecorder.addEventListener("dataavailable", event => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                });

                mediaRecorder.addEventListener("stop", handleRecordingStop);
                mediaRecorder.start();

                isRecording = true;
                recordBtn.textContent = 'Stop Recording';
                recordBtn.classList.add('recording', 'bg-gray-600', 'hover:bg-gray-700');
                recordBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
                statusDiv.textContent = 'Recording...';
                
                resultsDiv.classList.add('section-hidden');
                resultsDiv.classList.remove('section-visible');
                queryAudioSection.classList.add('section-hidden');
                queryAudioSection.classList.remove('section-visible');
                downloadBtn.classList.add('hidden');
            })
            .catch(error => {
                console.error("Error accessing microphone:", error);
                statusDiv.textContent = 'Error: Could not access microphone.';
            });
    }
});

const handleRecordingStop = async () => {
    recordBtn.textContent = 'Start Recording';
    recordBtn.classList.remove('recording', 'bg-gray-600', 'hover:bg-gray-700');
    recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
    isRecording = false;
    statusDiv.textContent = 'Recording complete. Analyzing...';

    if (audioChunks.length === 0) {
        statusDiv.textContent = "Error: Recording captured no audio data. Is the mic working?";
        return;
    }

    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

    if (audioBlob.size === 0) {
        statusDiv.textContent = "Error: Recording failed. Please try again.";
        return;
    }

    if (recordedAudioUrl) {
        URL.revokeObjectURL(recordedAudioUrl);
    }
    recordedAudioUrl = URL.createObjectURL(audioBlob);

    downloadBtn.href = recordedAudioUrl;
    downloadBtn.download = `imitune-recording-${Date.now()}.webm`;
    downloadBtn.classList.remove('hidden');

    playBtn.disabled = true;
    playBtn.textContent = '...';

    // Make the container visible first
    queryAudioSection.classList.remove('section-hidden');
    queryAudioSection.classList.add('section-visible');
    
    // NOW that the container is visible, initialize WaveSurfer.
    initializeWaveSurfer();
    
    wavesurfer.load(recordedAudioUrl);

    const onnxArrayBuffer = await audioBlob.arrayBuffer();

    statusDiv.textContent = 'Extracting audio features...';
    const modelResult = await window.electronAPI.runOnnxModel(onnxArrayBuffer);

    if (!modelResult.success) {
        statusDiv.textContent = `Model error: ${modelResult.error}`;
        return;
    }

    statusDiv.textContent = 'Searching...';
    try {
        await new Promise(resolve => setTimeout(resolve, 1000));
        const fakeSearchResults = [
            { id: 'sound123', score: 0.98, name: 'Windy sound.wav', freesound_id: '341229' },
            { id: 'sound456', score: 0.95, name: 'Jet passing by.wav', freesound_id: '58758' },
            { id: 'sound789', score: 0.95, name: 'Ocean waves.wav', freesound_id: '13309' },
        ];

        displayResults(fakeSearchResults);

    } catch (error) {
        console.error('Backend error:', error);
        statusDiv.textContent = 'Error: Could not connect to the server.';
    }
};

function displayResults(results) {
    statusDiv.textContent = 'Search complete!';
    resultList.innerHTML = '';
    if (results.length === 0) {
        resultList.innerHTML = '<p class="text-slate-400">No results found.</p>';
    } else {
        results.forEach(result => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'p-4 bg-slate-800 rounded-lg flex items-center space-x-4';
            const iframeSrc = `https://freesound.org/embed/sound/iframe/${result.freesound_id}/simple/medium/`;
            itemDiv.innerHTML = `
                <div class="flex-grow">
                    <p class="font-bold text-white">${result.name}</p>
                    <p class="text-sm text-slate-400">Score: ${result.score}</p>
                    <iframe src="${iframeSrc}" width="100%" height="90" frameborder="0" scrolling="no"></iframe>
                </div>
            `;
            resultList.appendChild(itemDiv);
        });
    }

    resultsDiv.classList.remove('section-hidden');
    resultsDiv.classList.add('section-visible');
}

