import streamlit as st
import time
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

from model import mtcnn, model, device
from utils import get_embedding, get_embeddings_batch, compare

# Professional Page Config
st.set_page_config(page_title="CCTV Face Recognition", layout="wide", page_icon="👁️")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* Metric & Status Boxes */
    .status-box {
        padding: 20px;
        border-radius: 10px;
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
    }
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
    }
    hr {
        border-color: #30363d;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: 1px solid rgba(240, 246, 252, 0.1);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    /* Info text overrides */
    .st-emotion-cache-161b22 {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👁️ Enterprise CCTV Feed Dashboard")
st.markdown("Live facial recognition and security monitoring system. Upload a target to track across the camera feed.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    target_file = st.file_uploader("Upload Target Image", type=["jpg", "png", "jpeg"])
    threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6, 0.05)
    st.markdown("---")
    st.info("Upload an image of the person you want to track across the camera feed.")

target_emb = None

if target_file is not None:
    img = Image.open(target_file).convert('RGB')
    img_np = np.array(img)
    
    # Resize target image if it's too large to speed up initial detection
    if img_np.shape[1] > 1000:
        scale = 1000 / img_np.shape[1]
        img_np = cv2.resize(img_np, (0, 0), fx=scale, fy=scale)
        
    boxes, probs = mtcnn.detect(img_np)
    
    if boxes is None:
        st.sidebar.error("❌ No face detected in the uploaded image. Please try another.")
    else:
        faces = mtcnn.extract(img_np, boxes, save_path=None)
        if faces is not None:
            # Take the first detected face from the target image
            target_emb = get_embedding(faces[0])
            st.sidebar.success("✅ Target face loaded successfully.")
            
            # Display target face preview
            with st.sidebar.expander("Target Face Preview", expanded=True):
                st.image(img_np, use_container_width=True)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Performance optimization: Downscale the image for faster detection
        scale_factor = 0.5
        small_rgb = cv2.resize(rgb, (0, 0), fx=scale_factor, fy=scale_factor)
        
        import torch
        t_start = time.time()
        boxes, probs = mtcnn.detect(small_rgb)

        if boxes is not None:
            # Rescale boxes back to original image size
            boxes = boxes / scale_factor
            
            # Extract faces directly from the original resolution image
            faces = mtcnn.extract(rgb, boxes, save_path=None)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_detect = time.time()
            detection_ms = (t_detect - t_start) * 1000
            
            if faces is not None:
                if target_emb is not None:
                    embs = get_embeddings_batch(faces)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_embed = time.time()
                    embedding_ms = (t_embed - t_detect) * 1000
                else:
                    embs = [None] * len(faces)
                    embedding_ms = 0
                    
                total_ms = (time.time() - t_start) * 1000
                print(f"Detect: {detection_ms:.2f}ms | Embed: {embedding_ms:.2f}ms | Total: {total_ms:.2f}ms")

                for i, emb in enumerate(embs):
                    x1, y1, x2, y2 = boxes[i]
                    
                    if target_emb is not None:
                        sim = compare(target_emb, emb)

                        if sim > threshold:
                            color = (0, 255, 0) # Green for match
                            label = f"MATCH: {sim:.2f}"
                        else:
                            color = (0, 0, 255) # Red for unknown
                            label = f"UNKNOWN: {sim:.2f}"
                    else:
                        color = (255, 165, 0) # Orange for generic face detection
                        label = "FACE DETECTED"

                    thickness = 2 if target_emb is None else 3
                    
                    # Draw a pro-looking label background
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (int(x1), int(y1)-30), (int(x1)+w, int(y1)), color, -1)
                    
                    # Draw Label text
                    cv2.putText(img, label, (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2 = st.columns([3, 1], gap="large")

with col1:
    st.markdown("### 🎥 Live Feed")
    webrtc_streamer(
        key="cctv-face-recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
with col2:
    st.markdown("### 📊 System Status")
    st.markdown(f'''
        <div class="status-box">
            <b>Camera Link:</b> 🟢 Active<br>
            <hr>
            <b>Model Backbone:</b> InceptionResnetV1<br>
            <b>Detection:</b> MTCNN<br>
            <b>Device:</b> {str(device).upper()}<br>
            <hr>
            <b>Latency Check:</b> OK
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### ℹ️ Instructions")
    st.info('''
    1. **Allow camera permissions** when prompted by your browser.
    2. **Upload a target image** of a person in the sidebar.
    3. **Adjust the threshold** to tune recognition sensitivity (0.6 is recommended).
    4. **Stand in front** of the camera to test the real-time CCTV recognition loop.
    ''')