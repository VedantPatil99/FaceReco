import time
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from model import face_detector as mtcnn
from sklearn.metrics.pairwise import cosine_similarity
import sys

def get_dummy_frame(width=640, height=480):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

def run_performance_metrics(num_frames=100):
    # Redirect stdout to both console and file
    log_file = open("utf8_metrics.txt", "w", encoding="utf-8")
    
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"System Device: {str(device).upper()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize Models
    log("Initializing models...")
    # mtcnn (RetinaFace) is already initialized and imported
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Warm-up
    log("Warming up models...")
    dummy_frame = get_dummy_frame()
    for _ in range(5):
        _ = mtcnn.detect(dummy_frame)
        dummy_face = torch.randn(1, 3, 160, 160).to(device)
        with torch.no_grad():
            _ = model(dummy_face)
    
    results = {
        'Detection (RetinaFace)': [],
        'Extraction (RetinaFace)': [],
        'Inference Single (ResNet)': [],
        'Inference Batch 5 (ResNet)': [],
        'Inference Batch 10 (ResNet)': [],
        'Data Transfer (to GPU)': [],
        'Similarity Match': [],
        'Frame Drawing (CV2)': []
    }
    
    dummy_boxes = np.array([[100, 100, 260, 260]])
    emb1 = np.random.rand(512)
    emb2 = np.random.rand(512)
    
    log(f"Running {num_frames} iterations for all metrics...")
    
    for i in range(num_frames):
        frame = get_dummy_frame()
        
        # 1. Detection
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        boxes, _ = mtcnn.detect(frame)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        results['Detection (RetinaFace)'].append((time.perf_counter() - t0) * 1000)
        
        # 2. Extraction
        t0 = time.perf_counter()
        faces = mtcnn.extract(frame, dummy_boxes, save_path=None)
        results['Extraction (RetinaFace)'].append((time.perf_counter() - t0) * 1000)
        
        # 3. Data Transfer
        t0 = time.perf_counter()
        if faces is not None:
            face_tensor = faces.to(device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        results['Data Transfer (to GPU)'].append((time.perf_counter() - t0) * 1000)
        
        # 4. Single Inference
        t0 = time.perf_counter()
        if faces is not None:
            with torch.no_grad():
                _ = model(face_tensor)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        results['Inference Single (ResNet)'].append((time.perf_counter() - t0) * 1000)
        
        # 5. Batch Inference (5)
        batch5 = torch.randn(5, 3, 160, 160).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batch5)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        results['Inference Batch 5 (ResNet)'].append((time.perf_counter() - t0) * 1000)
        
        # 6. Batch Inference (10)
        batch10 = torch.randn(10, 3, 160, 160).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batch10)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        results['Inference Batch 10 (ResNet)'].append((time.perf_counter() - t0) * 1000)
        
        # 7. Similarity Math
        t0 = time.perf_counter()
        _ = cosine_similarity([emb1], [emb2])[0][0]
        results['Similarity Match'].append((time.perf_counter() - t0) * 1000)
        
        # 8. Drawing
        t0 = time.perf_counter()
        cv2.rectangle(frame, (100, 100), (260, 260), (0, 255, 0), 2)
        cv2.putText(frame, "TARGET: 0.99", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        results['Frame Drawing (CV2)'].append((time.perf_counter() - t0) * 1000)

    total_latencies = []
    for j in range(num_frames):
        lat = (results['Detection (RetinaFace)'][j] + 
               results['Extraction (RetinaFace)'][j] + 
               results['Data Transfer (to GPU)'][j] + 
               results['Inference Single (ResNet)'][j] + 
               results['Similarity Match'][j] + 
               results['Frame Drawing (CV2)'][j])
        total_latencies.append(lat)
        
    avg_total = np.mean(total_latencies)
    fps = 1000 / avg_total
    
    log("\n" + "="*60)
    log(f"{'OPERATION':<30} | {'MEAN (ms)':<10} | {'STD (ms)':<10}")
    log("-" * 60)
    for op, times in results.items():
        log(f"{op:<30} | {np.mean(times):<10.2f} | {np.std(times):<10.2f}")
    
    log("="*60)
    log(f"TOTAL PIPELINE LATENCY: {avg_total:.2f} ms")
    log(f"ESTIMATED REAL-TIME FPS: {fps:.2f}")
    log("="*60)
    
    log_file.close()

if __name__ == '__main__':
    run_performance_metrics(100)