import tkinter as tk
import cv2
from PIL import Image, ImageTk

class SafeDriveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SafeDrive Monitor - Live Vision")
        self.root.geometry("800x600")
        
        # 1. Add a title
        self.title_label = tk.Label(root, text="SafeDrive Monitor: Live Feed", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=10)
        
        # 2. Add a label where the video will be displayed
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
        # 3. Turn on the webcam (0 is usually your laptop's default camera)
        self.cap = cv2.VideoCapture(0) 
        
        # 4. Start the video loop
        self.update_frame()

    def update_frame(self):
        # Read a frame from the webcam
        ret, frame = self.cap.read()
        
        if ret:
            # OpenCV uses BGR colors, but Tkinter needs RGB. Let's convert it.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert the frame to an image Tkinter can use
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Tell the app to run this function again in 15 milliseconds (creates the video effect)
        self.root.after(15, self.update_frame)
        
    def on_closing(self):
        # When you close the app, release the webcam so other apps can use it
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SafeDriveApp(root)
    # Ensure the camera turns off when the window is closed
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()