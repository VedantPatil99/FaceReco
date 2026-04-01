import cv2
from src.utils import get_embedding, compare_embeddings

def live_recognition(target_emb, threshold=0.6):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emb = get_embedding(frame)

        if emb is not None:
            sim = compare_embeddings(target_emb, emb)

            if sim > threshold:
                text = f"MATCH {sim:.2f}"
                color = (0,255,0)
            else:
                text = f"NO MATCH {sim:.2f}"
                color = (0,0,255)

            cv2.putText(frame, text, (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Live Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()