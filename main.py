import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face):
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face)
    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb

target_img = cv2.imread("target.jpg")
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

target_faces = mtcnn(target_img)
target_emb = get_embedding(target_faces[0])

cap = cv2.VideoCapture(0)

threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, faces = mtcnn.detect(rgb), mtcnn(rgb)

    if faces is not None:
        for i, face in enumerate(faces):
            emb = get_embedding(face)
            sim = cosine_similarity([target_emb], [emb])[0][0]

            x1, y1, x2, y2 = boxes[0][i]

            if sim > threshold:
                color = (0,255,0)
                label = f"MATCH {sim:.2f}"
            else:
                color = (0,0,255)
                label = f"{sim:.2f}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("CCTV Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()