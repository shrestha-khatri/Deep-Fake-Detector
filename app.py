import streamlit as st
import os
from backend.extract_frames import extract_frames
from backend.face_detect import extract_face
from backend.embed_faces import get_embedding
from backend.predict import predict

st.title("Deepfake Video Detector")

video = st.file_uploader("Upload video", type=["mp4"])

if video:
    with open("test.mp4", "wb") as f:
        f.write(video.read())

    extract_frames("test.mp4", "test_frames")

    embeddings = []
    for frame in os.listdir("test_frames"):
        if extract_face(
            f"test_frames/{frame}",
            f"test_faces/{frame}"
        ):
            embeddings.append(get_embedding(f"test_faces/{frame}"))

    avg_embedding = sum(embeddings) / len(embeddings)
    result, confidence = predict(avg_embedding)

    st.success(f"Prediction: {result}")
    st.info(f"Confidence: {confidence:.2f}")
