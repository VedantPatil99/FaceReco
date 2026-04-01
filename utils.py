import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from model import mtcnn, model, device

def get_embedding(face):
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(face)
    emb = emb.cpu().numpy()[0]
    emb = emb / np.linalg.norm(emb)
    return emb

def compare(e1, e2):
    return cosine_similarity([e1], [e2])[0][0]