"""Local SafeDrive Monitor dashboard using the existing FL3D checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from live_inference import (
    DriverAlertPolicy,
    ROOT,
    StateSoundPolicy,
    apply_eye_consistency_gate,
    apply_mouth_consistency_gate,
    eye_aspect_ratio,
    load_driver_state_model,
    mouth_aperture_ratio,
    prepare_face_tensor,
)
from models.driver_state_cnn import ID_TO_CLASS

UI_DIR = ROOT / "ui"
CHECKPOINT = ROOT / "weights" / "driver_state_cnn.pth"
MAX_FRAME_BYTES = 2_500_000
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class EyeClosurePolicy:
    """Timed EAR heuristics for gradual narrowing and prolonged closure."""

    NARROWING_THRESHOLD = 0.18
    NARROWING_HOLD_SECONDS = 0.7
    DEEP_CLOSURE_THRESHOLD = 0.16
    PROLONGED_SECONDS = 1.5
    MAX_GAP_SECONDS = 0.5
    REPEAT_WARNING_SECONDS = 4.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.narrowing_since = None
        self.deep_closure_since = None
        self.last_seen = None
        self.last_warning = None

    def update(self, eye_ratio: float, now: float) -> dict:
        if not np.isfinite(eye_ratio) or eye_ratio < 0:
            self.reset()
            return {"narrowing_active": False, "prolonged": False,
                    "narrowing_seconds": 0.0, "closed_seconds": 0.0, "play_urgent": False}
        if self.last_seen is not None and now - self.last_seen > self.MAX_GAP_SECONDS:
            self.narrowing_since = None
            self.deep_closure_since = None
            self.last_warning = None
        self.last_seen = now

        if eye_ratio < self.NARROWING_THRESHOLD:
            if self.narrowing_since is None:
                self.narrowing_since = now
        else:
            self.narrowing_since = None
        if eye_ratio < self.DEEP_CLOSURE_THRESHOLD:
            if self.deep_closure_since is None:
                self.deep_closure_since = now
        else:
            self.deep_closure_since = None
            self.last_warning = None

        narrowing_seconds = max(0.0, now - self.narrowing_since) if self.narrowing_since is not None else 0.0
        closed_seconds = max(0.0, now - self.deep_closure_since) if self.deep_closure_since is not None else 0.0
        prolonged = closed_seconds >= self.PROLONGED_SECONDS
        play_urgent = prolonged and (
            self.last_warning is None or now - self.last_warning >= self.REPEAT_WARNING_SECONDS
        )
        if play_urgent:
            self.last_warning = now
        return {
            "narrowing_active": narrowing_seconds >= self.NARROWING_HOLD_SECONDS,
            "prolonged": prolonged,
            "narrowing_seconds": narrowing_seconds,
            "closed_seconds": closed_seconds,
            "play_urgent": play_urgent,
        }


class InferenceEngine:
    """Own the model and one face landmarker; frame data is never written to disk."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_driver_state_model(CHECKPOINT, self.device)
        self.face_mesh = None
        self.face_landmarker = None
        if hasattr(mp, "solutions"):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        elif hasattr(mp, "tasks"):
            self.face_landmarker = self._create_tasks_landmarker()
        else:
            raise RuntimeError(f"MediaPipe {mp.__version__} has no supported face-landmark API.")
        self.lock = threading.Lock()
        self.score_history: deque[np.ndarray] = deque(maxlen=9)
        self.alert_policy = DriverAlertPolicy()
        self.eye_policy = EyeClosurePolicy()
        self._last_display_state = None
        self.sound_policy = StateSoundPolicy()
        self.frame_count = 0

    @staticmethod
    def _create_tasks_landmarker():
        """Use the maintained Tasks API when MediaPipe no longer bundles solutions."""
        if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
            cache_dir = Path(os.environ["LOCALAPPDATA"]) / "SafeDriveMonitor"
        else:
            cache_dir = Path.home() / ".cache" / "safedrive-monitor"
        model_path = cache_dir / "face_landmarker.task"
        if not model_path.is_file():
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = model_path.with_suffix(".download")
            try:
                print("Downloading the small MediaPipe face-landmarker model (about 4 MB)…")
                with urllib.request.urlopen(FACE_LANDMARKER_URL, timeout=30) as response:
                    with temporary_path.open("wb") as output:
                        shutil.copyfileobj(response, output)
                if temporary_path.stat().st_size < 1_000_000:
                    raise RuntimeError("The downloaded face-landmarker file is incomplete.")
                temporary_path.replace(model_path)
            except Exception as exc:
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"The small face-landmarker model is missing and could not be downloaded ({exc}). "
                    "Connect to the internet once, then restart SafeDrive."
                ) from None
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _find_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.face_mesh is not None:
            result = self.face_mesh.process(rgb)
            return result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(image)
        return result.face_landmarks[0] if result.face_landmarks else None

    @staticmethod
    def _reset_policy(policy: DriverAlertPolicy) -> None:
        policy.update(0.0, 1.0, 0.12484, time.monotonic())

    def process(self, encoded_frame: bytes) -> dict:
        if not encoded_frame or len(encoded_frame) > MAX_FRAME_BYTES:
            raise ValueError("Frame is empty or larger than the 2.5 MB limit.")
        with self.lock:
            raw = np.frombuffer(encoded_frame, dtype=np.uint8)
            frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("The browser frame could not be decoded.")
            landmarks = self._find_landmarks(frame)
            if landmarks is None:
                self.score_history.clear()
                self._reset_policy(self.alert_policy)
                self.eye_policy.reset()
                self._last_display_state = None
                self.sound_policy.last_state = None
                self.sound_policy.last_sound_at = None
                return {"status": "face_not_found", "state": "uncertain", "message": "Move into the camera view"}

            try:
                tensor = prepare_face_tensor(
                    frame, landmarks, self.model.normalization_mean, self.model.normalization_std
                ).to(self.device)
                with torch.inference_mode():
                    logits = self.model(tensor)[0] + self.model.decision_bias
                    scores = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
                eye_ratio = eye_aspect_ratio(landmarks, frame.shape[1], frame.shape[0])
                mouth_ratio = mouth_aperture_ratio(landmarks, frame.shape[1], frame.shape[0])
            except (ValueError, cv2.error, FloatingPointError) as exc:
                self.score_history.clear()
                self._reset_policy(self.alert_policy)
                self.eye_policy.reset()
                self._last_display_state = None
                self.sound_policy.last_state = None
                return {"status": "feature_error", "state": "uncertain", "message": str(exc)}

            self.score_history.append(scores)
            averaged = np.mean(np.stack(self.score_history), axis=0)
            raw_state = ID_TO_CLASS[str(int(np.argmax(averaged)))]
            state = apply_mouth_consistency_gate(
                raw_state, mouth_ratio, self.model.mouth_open_ratio_threshold
            )
            state = apply_eye_consistency_gate(
                state, eye_ratio, self.model.eye_closed_ratio_threshold
            )
            now = time.monotonic()
            warning_active, play_urgent = self.alert_policy.update(
                float(averaged[1]), eye_ratio, self.model.eye_closed_ratio_threshold, now
            )
            eye_result = self.eye_policy.update(eye_ratio, now)
            display_state = (
                "microsleep" if warning_active else
                "prolonged_closure" if eye_result["prolonged"] else
                "eye_closing" if eye_result["narrowing_active"] else state
            )
            warning_visible = warning_active or eye_result["prolonged"]
            cue_state = "microsleep" if warning_visible else display_state
            cue = self.sound_policy.update(cue_state, now)
            sound = "microsleep" if (play_urgent or eye_result["play_urgent"]) else (
                "eye_closing" if display_state == "eye_closing" and self._last_display_state != "eye_closing" else
                cue if cue in {"yawning", "uncertain"} else None
            )
            self._last_display_state = display_state
            self.frame_count += 1
            return {
                "status": "ok",
                "state": display_state,
                "raw_state": raw_state,
                "warning_active": warning_visible,
                "warning_reason": "model_and_eye" if warning_active else "prolonged_eye_closure" if eye_result["prolonged"] else None,
                "scores": {ID_TO_CLASS[str(i)]: float(averaged[i]) for i in range(3)},
                "eye_ratio": eye_ratio,
                "eye_threshold": float(self.model.eye_closed_ratio_threshold),
                "eye_narrowing_threshold": self.eye_policy.NARROWING_THRESHOLD,
                "eye_narrowing_seconds": eye_result["narrowing_seconds"],
                "eye_narrowing_active": eye_result["narrowing_active"],
                "eye_deep_closure_threshold": self.eye_policy.DEEP_CLOSURE_THRESHOLD,
                "eye_closed_seconds": eye_result["closed_seconds"],
                "prolonged_eye_closure": eye_result["prolonged"],
                "mouth_ratio": mouth_ratio,
                "mouth_threshold": float(self.model.mouth_open_ratio_threshold),
                "score_window": len(self.score_history),
                "frame_count": self.frame_count,
                "sound": sound,
                "message": "Sustained eye and model evidence detected" if warning_active else "Prolonged eye closure detected" if eye_result["prolonged"] else "",
            }

    def close(self) -> None:
        with self.lock:
            if self.face_mesh is not None:
                self.face_mesh.close()
            if self.face_landmarker is not None:
                self.face_landmarker.close()

    def reset(self) -> None:
        with self.lock:
            self.score_history.clear()
            self._reset_policy(self.alert_policy)
            self.eye_policy.reset()
            self._last_display_state = None
            self.sound_policy.last_state = None
            self.sound_policy.last_sound_at = None


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "SafeDriveLocal/1.0"

    def log_message(self, _format: str, *_args) -> None:
        # Avoid writing request details or camera session activity to terminal logs.
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, body: dict) -> None:
        self._send(status, json.dumps(body).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if self.path in {"/api/status", "/api/health"}:
            self._json(200, {
                "ready": True,
                "model": CHECKPOINT.name,
                "device": str(self.server.engine.device),
                "warning_policy": "model agreement (0.90 score + closed-eye cue for 1 second), plus EAR heuristics: narrowing below 0.18 for 0.7 seconds and prolonged closure below 0.16 for 1.5 seconds",
                "privacy": "Frames are processed locally in memory and are not saved.",
            })
            return
        route = routes.get(self.path)
        if route is None:
            self._json(404, {"error": "Not found"})
            return
        filename, content_type = route
        try:
            body = (UI_DIR / filename).read_bytes()
        except OSError:
            self._json(500, {"error": "Dashboard asset missing."})
            return
        self._send(200, body, content_type)

    def do_POST(self) -> None:
        if self.path == "/api/reset":
            self.server.engine.reset()
            self._json(200, {"reset": True})
            return
        if self.path != "/api/frame":
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_FRAME_BYTES:
            self._json(413, {"error": "Frame must be between 1 byte and 2.5 MB."})
            return
        try:
            result = self.server.engine.process(self.rfile.read(length))
        except (ValueError, cv2.error) as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception:
            self._json(500, {"error": "Inference failed. Stop and restart the local monitor."})
            return
        self._json(200, result)


class SafeDriveServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, engine):
        super().__init__(address, handler)
        self.engine = engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Local bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the dashboard automatically")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("The local dashboard only binds to this computer for privacy.")
    if not CHECKPOINT.is_file():
        parser.error(f"Missing trained checkpoint: {CHECKPOINT}")
    if not (UI_DIR / "index.html").is_file():
        parser.error(f"Missing dashboard assets: {UI_DIR}")

    try:
        engine = InferenceEngine()
    except Exception as exc:
        parser.error(f"Could not start the existing model: {exc}")
    server = SafeDriveServer((args.host, args.port), DashboardHandler, engine)
    url = f"http://{args.host}:{args.port}"
    print(f"SafeDrive Monitor is ready at {url}")
    print(f"Model: {CHECKPOINT.name} | device: {engine.device}")
    print("Local only. Camera frames are processed in memory and are not saved.")
    print("Press Ctrl+C here to stop the monitor.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url, new=1)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping SafeDrive Monitor.")
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
