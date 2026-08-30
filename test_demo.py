import cv2
from vision.extractor import VisionExtractor

def run_demo():
    extractor = VisionExtractor()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        metrics = extractor.process_frame(frame)
        
        if metrics:
            print(f"EAR: {metrics.ear:.2f} | MAR: {metrics.mar:.2f} | Pitch: {metrics.pitch:.2f}")
            
        cv2.imshow('SafeDrive Prototype', frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()