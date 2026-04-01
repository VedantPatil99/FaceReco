import cv2
import time
from model import mtcnn
from utils import get_embedding, compare

def live_recognition(target_emb, threshold=0.6, scale_factor=0.5):
    """
    Live facial recognition loop intended to be called by an external CCTV manager.
    Displays a specialized CCTV viewer.
    """
    cap = cv2.VideoCapture(0)

    print("CCTV Feed live. Press ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video feed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Performance scaling for detector only
        small_rgb = cv2.resize(rgb, (0, 0), fx=scale_factor, fy=scale_factor)
        boxes, probs = mtcnn.detect(small_rgb)

        target_detected = False

        if boxes is not None:
            # Scale boxes back
            boxes = boxes / scale_factor
            
            faces = mtcnn.extract(rgb, boxes, save_path=None)
            
            if faces is not None:
                for i, face in enumerate(faces):
                    emb = get_embedding(face)
                    sim = compare(target_emb, emb)
                    
                    x1, y1, x2, y2 = boxes[i]

                    if sim > threshold:
                        text = f"TARGET MATCH: {sim:.2f}"
                        color = (0, 255, 0)
                        target_detected = True
                        thickness = 3
                    else:
                        text = f"UNKNOWN: {sim:.2f}"
                        color = (0, 0, 255)
                        thickness = 2
                    
                    # Pro Label Box
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)-35), (int(x1)+w, int(y1)), color, -1)
                    
                    cv2.putText(frame, text, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
        # CCTV Overhead Status
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CCTV CAM 1 - {timestamp}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        # System Warning UI if Target Detected
        if target_detected:
            cv2.putText(frame, "!!! TARGET ACQUIRED !!!", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Live Face Recognition Node", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()