import torch
import clip
from PIL import Image
import os
import numpy as np

def get_clip_embeddings(folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    embeddings = []
    for fname in os.listdir(folder):
        if fname.endswith(".jpg"):
            image = preprocess(Image.open(os.path.join(folder, fname))).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image).cpu().numpy()[0]
            embeddings.append(embedding)
    return np.array(embeddings)

def diversity_score(embeddings):
    norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarity_matrix = norms @ norms.T
    upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    return 1 - np.mean(upper_tri)