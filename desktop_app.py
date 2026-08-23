import tkinter as tk
import cv2
import time
import threading
from PIL import Image, ImageTk

# Import our newly created backend modules
from database import DatabaseManager
from vision_model import VisionPredictor

class SafeDriveApp:
    def __init__(self, root) -> None:
        self.root = root
        self.root.title("SafeDrive Monitor - Real-Time Inference")
        self.root.geometry("900x700")
        
        # 1. Initialize Core Persistence and ML Modules
        self.db = DatabaseManager()
        self.vision = VisionPredictor()
        
        # 2. Application State
        self.current_frame = None
        self.status = "INITIALIZING"
        self.confidence = 0.0
        self.is_running = True

        self._build_ui()
        
        # 3. Hardware Initialization
        self.cap = cv2.VideoCapture(0)
        
        # 4. Threading: Run prediction engine in background to prevent UI blocking
        self.inference_thread = threading.Thread(target=self._run_inference, daemon=True)
        self.inference_thread.start()
        
        self._update_video_feed()

    def _build_ui(self) -> None:
        self.title_label = tk.Label(self.root, text="SafeDrive Monitor", font=("Helvetica", 24, "bold"))
        self.title_label.pack(pady=10)
        
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)
        
        # Status indicators for color-coded alerts
        self.status_label = tk.Label(self.root, text="Status: WAITING", font=("Helvetica", 18, "bold"), fg="white", bg="gray")
        self.status_label.pack(pady=10, fill=tk.X)

    def _run_inference(self) -> None:
        """Background thread evaluating frames for prediction and logging."""
        while self.is_running:
            if self.current_frame is not None:
                try:
                    # Execute Vision Model inference
                    self.status, self.confidence = self.vision.predict(self.current_frame)
                    
                    # Log to SQLite database for post-hoc analysis
                    self.db.log_prediction(self.status, self.confidence)
                    
                    # Update UI color-coding based on alert thresholds
                    if self.status == "NORMAL":
                        color = "green"
                    elif self.status == "DROWSY":
                        color = "orange"
                    else:
                        color = "red"
                        
                    self.status_label.config(text=f"Status: {self.status} (Conf: {self.confidence:.2f})", bg=color)
                except Exception as e:
                    print(f"Inference execution failed: {e}")
            
            # Throttling inference to ~2 FPS to reduce CPU load 
            time.sleep(0.5)

    def _update_video_feed(self) -> None:
        """Main thread loop for rendering the camera feed."""
        ret, frame = self.cap.read()
        
        if ret:
            # Store raw frame for the inference thread
            self.current_frame = frame.copy()
            
            # Convert frame for Tkinter rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        if self.is_running:
            self.root.after(30, self._update_video_feed)

    def on_closing(self) -> None:
        """Graceful shutdown protocol."""
        self.is_running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SafeDriveApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()