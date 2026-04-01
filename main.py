import cv2
import os
import torch
import numpy as np
import argparse

from model import mtcnn, device
from utils import get_embedding, compare

def main(target_image_path, threshold=0.6):
    if not os.path.exists(target_image_path):
        print(f"Error: Target image '{target_image_path}' not found.")
        return

    # Load and process target image
    print(f"Loading target image from {target_image_path}...")
    target_img = cv2.imread(target_image_path)
    target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    # Detect faces in Target Image
    boxes, probs = mtcnn.detect(target_img_rgb)
    if boxes is None:
        print("Error: No face detected in the target image.")
        return
        
    # Extract the first face
    faces = mtcnn.extract(target_img_rgb, boxes, save_path=None)
    if faces is None:
        print("Error: Could not extract face from the target image.")
        return
        
    target_emb = get_embedding(faces[0])
    print("Target face loaded successfully. Starting CCTV camera feed...")

    cap = cv2.VideoCapture(0)
    
    # Optional performance optimization parameters
    scale_factor = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Downscale for faster MTCNN detection
        small_rgb = cv2.resize(rgb, (0, 0), fx=scale_factor, fy=scale_factor)
        boxes, probs = mtcnn.detect(small_rgb)

        if boxes is not None:
            # Rescale boxes back
            boxes = boxes / scale_factor
            
            # Extract faces directly from the original resolution image
            faces_batch = mtcnn.extract(rgb, boxes, save_path=None)
            
            if faces_batch is not None:
                for i, face in enumerate(faces_batch):
                    emb = get_embedding(face)
                    sim = compare(target_emb, emb)

                    x1, y1, x2, y2 = boxes[i]

                    # Draw Boxes and Labels
                    if sim > threshold:
                        color = (0, 255, 0)
                        label = f"MATCH: {sim:.2f}"
                        thickness = 3
                    else:
                        color = (0, 0, 255)
                        label = f"UNKNOWN: {sim:.2f}"
                        thickness = 2

                    # Background rectangle for text
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)-35), (int(x1)+w, int(y1)), color, -1)
                    
                    # Text and Face Box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add CCTV overlay text
        cv2.putText(frame, "ENTERPRISE CCTV LIVE FEED", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Press ESC to exit", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("CCTV Face Recognition Window", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live CCTV Facial Recognition via Webcam")
    parser.add_argument("--image", type=str, default="target.jpg", help="Path to the target face image.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold (default: 0.6).")
    args = parser.parse_args()
    
    main(args.image, args.threshold)