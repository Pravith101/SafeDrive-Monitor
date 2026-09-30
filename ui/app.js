const $ = (id) => document.getElementById(id);
const video = $('camera');
const capture = $('capture');
const stage = $('camera-stage');
let stream = null;
let loopTimer = null;
let sessionStartedAt = 0;
let soundContext = null;
let soundsEnabled = false;
let lastState = 'idle';
let toastTimer = null;
let busy = false;

const labels = {
  alert: { title: 'Alert', kicker: 'NO TARGET CUE DETECTED', description: 'The current frames resemble the dataset’s alert class. This reading cannot establish that you are safe to continue.', code: 'STATE 01' },
  yawning: { title: 'Yawning cue', kicker: 'MOUTH CUE AGREES', description: 'The model and mouth-opening measurement agree. Take this as a cue to check how you feel.', code: 'STATE 02' },
  uncertain: { title: 'Reading uncertain', kicker: 'EVIDENCE DOES NOT AGREE', description: 'The face or model cues are unclear. Adjust the camera view and do not treat this as an alert reading.', code: 'CHECK VIEW' },
  eye_closing: { title: 'Eyes narrowing', kicker: 'SUSTAINED EYELID CHANGE', description: 'The eye-opening measurement has stayed low for a short period. Check how you feel and take a break if you are tired.', code: 'EARLY CUE' },
  prolonged_closure: { title: 'Prolonged eye closure', kicker: 'EYES APPEAR CLOSED', description: 'The eye-opening measurement has remained very low. This may indicate a prolonged closure; it cannot confirm that someone is asleep. Pull over safely if drowsy.', code: 'URGENT' },
  microsleep: { title: 'Microsleep warning', kicker: 'SUSTAINED EVIDENCE DETECTED', description: 'The model score and eye-closure cue have agreed continuously. Pull over safely as soon as possible.', code: 'URGENT' },
  idle: { title: 'Monitoring is off', kicker: 'READY WHEN YOU ARE', description: 'Start the camera monitor when parked and ready. This prototype cannot determine whether it is safe to drive.', code: 'IDLE' },
  face_not_found: { title: 'Face not in view', kicker: 'TRACKING PAUSED', description: 'Move into the camera view. The monitor will resume when it can see a face.', code: 'NO FACE' },
  feature_error: { title: 'Reading unavailable', kicker: 'FEATURES NOT RELIABLE', description: 'The face landmarks could not be measured clearly. Reposition the camera and try again.', code: 'CHECK VIEW' }
};

function showToast(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3800);
}

function setState(state, warning = false) {
  const info = labels[state] || labels.uncertain;
  const card = $('state-card');
  card.className = `state-card state-${state}`;
  $('state-code').textContent = info.code;
  $('state-kicker').textContent = info.kicker;
  $('state-title').textContent = info.title;
  $('state-description').textContent = info.description;
  $('warning-banner').hidden = !warning;
  if (warning) {
    const prolonged = state === 'prolonged_closure';
    $('warning-banner').querySelector('strong').textContent = prolonged ? 'PROLONGED EYE CLOSURE' : 'DROWSINESS WARNING';
    $('warning-banner').querySelector('span:last-child').textContent = prolonged
      ? 'Eyes appear closed. Pull over safely if you feel drowsy.'
      : 'Model and eye cues agree. Pull over safely as soon as you can.';
    $('warning-banner').classList.toggle('warning-closure', prolonged);
  }
  const nextState = warning ? 'microsleep' : state;
  if (nextState !== lastState && nextState !== 'idle' && nextState !== 'face_not_found') {
    addEvent(nextState);
  }
  lastState = nextState;
}

function addEvent(state) {
  // Event changes are announced accessibly through the main state card.
  const name = (labels[state] || labels.uncertain).title;
  $('camera-hint').textContent = `${name} · ${new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'})}`;
}

function setCameraStatus(active, text) {
  $('camera-status').textContent = text;
  $('camera-indicator').classList.toggle('is-live', active);
  $('live-tag').classList.toggle('is-live', active);
  $('live-tag').innerHTML = `<span></span> ${active ? 'LIVE' : 'STANDBY'}`;
  $('camera-stage').classList.toggle('is-live', active);
}

function percent(value) {
  return `${Math.round(Math.max(0, Math.min(1, value || 0)) * 100)}%`;
}

function renderResult(data, elapsedMs) {
  if (data.status === 'face_not_found') {
    $('face-tag').hidden = true;
    setState('face_not_found');
    $('camera-hint').textContent = data.message;
    resetCueReadouts();
    return;
  }
  if (data.status === 'feature_error') {
    $('face-tag').hidden = true;
    setState('feature_error');
    $('camera-hint').textContent = data.message;
    return;
  }
  $('face-tag').hidden = false;
  $('camera-hint').textContent = `Inference ${elapsedMs} ms · ${data.score_window}/9 frames smoothed`;
  $('frame-counter').textContent = `${data.frame_count.toLocaleString()} frames`;
  const score = data.scores.microsleep;
  $('score-value').textContent = percent(score);
  $('score-meter').style.width = percent(score);
  $('score-meter').classList.toggle('is-hot', score >= 0.9);
  $('eye-value').textContent = Number(data.eye_ratio).toFixed(3);
  $('eye-threshold').textContent = `${Number(data.eye_narrowing_threshold).toFixed(2)} / ${Number(data.eye_deep_closure_threshold).toFixed(2)}`;
  $('mouth-value').textContent = Number(data.mouth_ratio).toFixed(3);
  $('mouth-threshold').textContent = Number(data.mouth_threshold).toFixed(3);
  const eyeClosed = data.eye_ratio < data.eye_threshold;
  const mouthOpen = data.mouth_ratio >= data.mouth_threshold;
  const prolonged = Boolean(data.prolonged_eye_closure);
  const narrowing = Boolean(data.eye_narrowing_active);
  $('eye-state').textContent = prolonged ? `CLOSED ${Number(data.eye_closed_seconds).toFixed(1)}s` : narrowing ? `NARROW ${Number(data.eye_narrowing_seconds).toFixed(1)}s` : eyeClosed ? 'CLOSED CUE' : 'OPEN CUE';
  $('eye-state').className = `cue-result ${prolonged || eyeClosed ? 'is-closed' : narrowing ? 'is-narrowing' : 'is-supported'}`;
  $('mouth-state').textContent = mouthOpen ? 'OPEN CUE' : 'RESTING';
  $('mouth-state').className = `cue-result ${mouthOpen ? 'is-supported' : ''}`;
  setState(data.state, data.warning_active);
  if (data.sound) playCue(data.sound);
  $('connection-label').textContent = `LOCAL INFERENCE · ${elapsedMs} MS`;
}

function resetCueReadouts() {
  $('score-value').textContent = '—';
  $('score-meter').style.width = '0';
  $('eye-value').textContent = '—';
  $('mouth-value').textContent = '—';
  $('eye-state').textContent = 'NO DATA';
  $('mouth-state').textContent = 'NO DATA';
  $('frame-counter').textContent = '0 frames';
}

async function analyzeFrame() {
  if (!stream || busy) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    loopTimer = setTimeout(analyzeFrame, 250);
    return;
  }
  busy = true;
  const maxWidth = 640;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  capture.width = Math.round(video.videoWidth * scale);
  capture.height = Math.round(video.videoHeight * scale);
  capture.getContext('2d', {alpha:false}).drawImage(video, 0, 0, capture.width, capture.height);
  const blob = await new Promise(resolve => capture.toBlob(resolve, 'image/jpeg', 0.78));
  const started = performance.now();
  try {
    const response = await fetch('/api/frame', {method:'POST', headers:{'Content-Type':'image/jpeg'}, body:blob, cache:'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Inference request failed.');
    const elapsed = Math.round(performance.now() - started);
    renderResult(data, elapsed);
  } catch (error) {
    $('connection-label').textContent = 'LOCAL CONNECTION INTERRUPTED';
    showToast(error.message || 'Could not reach the local inference service.');
  } finally {
    busy = false;
    if (stream) loopTimer = setTimeout(analyzeFrame, 280);
  }
}

async function startMonitoring() {
  if (stream) return;
  $('start-button').disabled = true;
  $('connection-label').textContent = 'REQUESTING CAMERA ACCESS';
  try {
    await fetch('/api/reset', {method:'POST'});
    stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:24,max:30}},audio:false});
    video.srcObject = stream;
    await video.play();
    sessionStartedAt = Date.now();
    $('stop-button').disabled = false;
    $('start-button').textContent = 'Monitoring active';
    setCameraStatus(true, 'CAMERA LIVE');
    $('connection-label').textContent = 'LOCAL INFERENCE READY';
    if (!soundsEnabled) showToast('Monitoring is active. Click “Sound off” in the header to enable audio cues.');
    loopTimer = setTimeout(analyzeFrame, 180);
  } catch (error) {
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    $('start-button').disabled = false;
    $('connection-label').textContent = 'CAMERA UNAVAILABLE';
    const detail = error.name === 'NotAllowedError' ? 'Camera permission was blocked. Allow camera access for this local page, then retry.' : error.name === 'NotFoundError' ? 'No camera was found. Connect a webcam and retry.' : 'The camera could not start. Close other camera apps and retry.';
    showToast(detail);
  }
}

async function stopMonitoring() {
  if (!stream) return;
  clearTimeout(loopTimer);
  stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.srcObject = null;
  busy = false;
  $('stop-button').disabled = true;
  $('start-button').disabled = false;
  $('start-button').innerHTML = '<span class="button-icon">▶</span> Start monitoring';
  $('face-tag').hidden = true;
  setCameraStatus(false, 'CAMERA OFF');
  setState('idle');
  resetCueReadouts();
  $('camera-hint').textContent = 'Position your face in the center of the frame.';
  $('connection-label').textContent = 'LOCAL SERVER READY';
  try { await fetch('/api/reset', {method:'POST'}); } catch (_) { /* local session is already stopped */ }
}

function playCue(kind) {
  if (!soundsEnabled || !soundContext) return;
  const patterns = {
    microsleep: [[990,180],[720,180],[990,310]],
    eye_closing: [[510,100],[650,120]],
    yawning: [[620,105],[850,155]],
    uncertain: [[420,145]]
  };
  const pattern = patterns[kind];
  if (!pattern) return;
  let startAt = soundContext.currentTime + 0.02;
  for (const [frequency, duration] of pattern) {
    const oscillator = soundContext.createOscillator();
    const gain = soundContext.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;
    gain.gain.setValueAtTime(0.0001, startAt);
    gain.gain.exponentialRampToValueAtTime(kind === 'microsleep' ? 0.17 : 0.095, startAt + 0.018);
    gain.gain.exponentialRampToValueAtTime(0.0001, startAt + duration / 1000);
    oscillator.connect(gain).connect(soundContext.destination);
    oscillator.start(startAt);
    oscillator.stop(startAt + duration / 1000 + 0.01);
    startAt += duration / 1000 + 0.1;
  }
}

async function toggleSound() {
  try {
    if (!soundContext) soundContext = new (window.AudioContext || window.webkitAudioContext)();
    if (soundsEnabled) {
      soundsEnabled = false;
      $('sound-toggle').setAttribute('aria-pressed','false');
      $('sound-toggle').querySelector('span').textContent = 'Sound off';
      showToast('Warning sounds muted. Visual warnings remain active.');
    } else {
      await soundContext.resume();
      soundsEnabled = true;
      $('sound-toggle').setAttribute('aria-pressed','true');
      $('sound-toggle').querySelector('span').textContent = 'Sound on';
      playCue('yawning');
      showToast('State cues enabled: soft for yawning, neutral for uncertain, urgent for sustained microsleep evidence.');
    }
  } catch (_) {
    showToast('This browser could not enable audio. Visual warnings are still available.');
  }
}

function updateClock() {
  if (!sessionStartedAt || !stream) return;
  const elapsed = Math.floor((Date.now() - sessionStartedAt) / 1000);
  $('session-clock').textContent = `${String(Math.floor(elapsed/60)).padStart(2,'0')}:${String(elapsed%60).padStart(2,'0')}`;
}

async function connect() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    const status = await response.json();
    if (!response.ok || !status.ready) throw new Error('Model is not ready.');
    $('device-label').textContent = status.device.toUpperCase();
    $('connection-label').textContent = 'LOCAL SERVER READY';
  } catch (_) {
    $('connection-label').textContent = 'MODEL CONNECTION ERROR';
    showToast('The local model service is unavailable. Restart SafeDrive from the project folder.');
  }
}

$('start-button').addEventListener('click', startMonitoring);
$('stop-button').addEventListener('click', stopMonitoring);
$('sound-toggle').addEventListener('click', toggleSound);
window.addEventListener('beforeunload', () => {
  if (stream) stream.getTracks().forEach(track => track.stop());
  fetch('/api/reset', {method:'POST', keepalive:true}).catch(() => {});
});
setInterval(updateClock, 500);
connect();
