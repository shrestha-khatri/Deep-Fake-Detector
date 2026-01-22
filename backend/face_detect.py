from facenet_pytorch import MTCNN
from PIL import Image
import os
import torch
import numpy as np

mtcnn = MTCNN(image_size=224, margin=20)

def extract_face(img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return False

    # face: torch tensor (3, H, W), float [0,1]
    face = face.permute(1, 2, 0)          # (H, W, 3)
    face = (face * 255).byte().numpy()    # uint8

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(face).save(save_path)

    return True
