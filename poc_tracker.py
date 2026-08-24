import cv2
import mediapipe as mp
import numpy as np

class FaceLandmarkDetector:
    """
    Extracts and visualizes spatial facial landmarks utilizing MediaPipe Face Mesh.
    Serves as the data-extraction pipeline for the SafeDrive CNN.
    """
    def __init__(self, max_faces: int = 1) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Styling for the mesh and isolated features
        self.mesh_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        self.contour_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the base tessellation (green mesh)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mesh_spec
                )
                # Highlight critical regions: Eyes, Eyebrows, Lips (red contours)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.contour_spec
                )
        return frame

def run_demo() -> None:
    """Initializes the webcam and executes the real-time tracking loop."""
    cap = cv2.VideoCapture(0)
    detector = FaceLandmarkDetector()

    print("Initializing PoC. Press 'q' in the video window to terminate.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from camera.")
            break

        annotated_frame = detector.process_frame(frame)
        
        # Render output
        cv2.imshow('SafeDrive Monitor - Review 1 PoC', annotated_frame)

        # Break loop on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()