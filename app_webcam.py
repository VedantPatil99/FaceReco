import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=device, image_size=160)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face):
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face)
    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb

st.title("Live Face Recognition (Webcam)")

target_file = st.file_uploader("Upload Target Image", type=["jpg","png"])
threshold = st.slider("Similarity Threshold", 0.3, 0.9, 0.6)

target_emb = None

if target_file:
    img = Image.open(target_file).convert('RGB')
    img = np.array(img)
    faces = mtcnn(img)
    
    if faces is None:
        st.error("No face detected in image")
    else:
        target_emb = get_embedding(faces[0])
        st.success("Target face loaded")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(rgb)
        faces = mtcnn(rgb)

        if faces is not None and boxes is not None and target_emb is not None:
            for i, face in enumerate(faces):
                emb = get_embedding(face)
                sim = cosine_similarity([target_emb], [emb])[0][0]

                x1, y1, x2, y2 = boxes[i]

                if sim > threshold:
                    color = (0,255,0)
                    label = f"MATCH {sim:.2f}"
                else:
                    color = (0,0,255)
                    label = f"{sim:.2f}"

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(img, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="face-recognition",
    video_processor_factory=VideoProcessor
)