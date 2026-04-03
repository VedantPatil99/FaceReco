# 👁️ FaceReco: Real-Time CCTV Face Recognition

## 📌 Overview
FaceReco is a high-performance, real-time facial recognition pipeline built for CCTV and webcam video streams. It detects and recognizes faces on the fly by comparing them against an uploaded target image, using hardware-accelerated deep learning models. 

The project includes an elegant **Streamlit WebRTC Dashboard** that lets users seamlessly start their camera, upload the target picture to track, and observe real-time tracking metrics. 

## 🚀 Key Features
- **Live Real-time Feed**: Stream webcam video directly to the browser using `streamlit-webrtc`.
- **High-Accuracy Pipelines**: Uses **MTCNN** (Multi-task Cascaded Convolutional Networks) for robust face detection and bounding box alignment.
- **Deep Feature Embedding**: Uses **InceptionResnetV1** for extracting 512-dimensional face embeddings.
- **GPU Acceleration**: Fully accelerated via PyTorch to leverage NVIDIA GPUs (CUDA) for extremely low-latency batch processing.
- **Dynamic Thresholding**: Adjust similarity thresholds dynamically on the UI to prevent false positives.
- **Benchmarking Tools**: Includes built-in metrics and benchmarking tools to empirically track frame-by-frame latency.

## 🛠️ Technology Stack
- **Deep Learning**: PyTorch, Facenet-PyTorch
- **Computer Vision**: OpenCV, Pillow
- **Frontend / UI**: Streamlit, Streamlit-WebRTC
- **Data & Processing**: NumPy, Pandas, Scikit-learn

## 📁 Project Structure
```text
📦 FaceReco
├── app_webcam.py       # Main Streamlit UI with the CCTV WEBRTC functionality
├── main.py             # CLI inference and simple demonstration script
├── metrics.py          # Script for empirical performance benchmarking and latency logging
├── model.py            # Model definitions and hardware bindings (PyTorch/CUDA)
├── recognize.py        # Logic for embeddings comparison and facial matching
├── utils.py            # Helper functions for bounding box management and vector math
├── requirements.txt    # Project dependencies
└── verify_gpu.py       # Script to check PyTorch + CUDA acceleration availability
```

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/VedantPatil99/FaceReco.git
   cd FaceReco
   ```

2. **Setup a Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install all the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage Guide

### 1. Launch the CCTV Dashboard
Run the Streamlit live-feed application:
```bash
streamlit run app_webcam.py
```
*Open `http://localhost:8501` in your browser. Give camera permissions, upload a picture of the target in the sidebar, and watch the system track the target in real-time.*

### 2. Verify GPU Configuration
If you have an NVIDIA GPU, verify if CUDA is correctly hooked up:
```bash
python verify_gpu.py
```

### 3. Run Latency Benchmarks
Evaluate the detection and embedding latency metrics on your hardware:
```bash
python metrics.py
```

## 📝 License
This project is for educational and research purposes.
