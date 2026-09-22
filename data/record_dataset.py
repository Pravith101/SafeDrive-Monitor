import cv2
import os
import sys
import numpy as np
import mediapipe as mp

# Add parent directory to path to import vision module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vision.extractor import VisionExtractor

def collect_data(sequence_length: int = 30) -> None:
    print("Initializing Webcam & MediaPipe...")
    cap = cv2.VideoCapture(0)
    extractor = VisionExtractor()
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.6
    )

    sequences = []
    labels = []
    current_sequence = []
    
    print("\n--- Data Collection Instructions ---")
    print("Press 'n' to record a NORMAL sequence (awake, attentive).")
    print("Press 'd' to record a DROWSY sequence (yawning, eyes closing, head nodding).")
    print("Press 's' to SAVE and EXIT.")
    
    recording_label = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(frame_rgb)
        
        # Display instructions on screen
        cv2.putText(frame, "Keys: [n] Normal | [d] Drowsy | [s] Save", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            img_h, img_w, _ = frame.shape
            
            # Use Sahil's extractor
            metrics = extractor.extract_metrics(landmarks, img_w, img_h)
            feature_vector = [metrics.ear, metrics.mar, metrics.pitch, metrics.yaw, metrics.roll]

            if recording_label is not None:
                current_sequence.append(feature_vector)
                cv2.putText(frame, f"RECORDING {recording_label.upper()}... {len(current_sequence)}/{sequence_length}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if len(current_sequence) == sequence_length:
                    sequences.append(current_sequence)
                    labels.append(0 if recording_label == 'normal' else 1)
                    print(f"Saved {recording_label} sequence! Total: {len(sequences)}")
                    current_sequence = []
                    recording_label = None
        else:
            if recording_label is not None:
                cv2.putText(frame, "FACE LOST - PAUSED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('SafeDrive Data Collector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n') and recording_label is None:
            recording_label = 'normal'
            current_sequence = []
        elif key == ord('d') and recording_label is None:
            recording_label = 'drowsy'
            current_sequence = []
        elif key == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if sequences:
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/sequences.npy', np.array(sequences, dtype=np.float32))
        np.save('data/processed/labels.npy', np.array(labels, dtype=np.int64))
        print(f"\nSuccessfully saved {len(sequences)} sequences to data/processed/")
    else:
        print("\nNo data recorded.")

if __name__ == "__main__":
    collect_data()