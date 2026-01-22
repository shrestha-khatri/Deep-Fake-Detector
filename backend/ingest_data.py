import os, uuid, shutil
from qdrant_client import QdrantClient
from extract_frames import extract_frames
from face_detect import extract_face
from embed_faces import get_embedding

client = QdrantClient(host="localhost", port=6333)
COLLECTION = "deepfake_faces"
DATA_DIR = "data"

for label in ["real", "fake"]:
    video_dir = os.path.join(DATA_DIR, label)

    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video)

        frames_dir = "temp_frames"
        faces_dir = "temp_faces"

        extract_frames(video_path, frames_dir)

        for frame in os.listdir(frames_dir):
            face_ok = extract_face(
                f"{frames_dir}/{frame}",
                f"{faces_dir}/{frame}"
            )
            if not face_ok:
                continue

            embedding = get_embedding(f"{faces_dir}/{frame}")

            client.upsert(
                collection_name=COLLECTION,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": embedding.tolist(),
                    "payload": {
                        "label": label,
                        "video": video
                    }
                }]
            )

        shutil.rmtree(frames_dir)
        shutil.rmtree(faces_dir)
        print(f"Ingested {video} [{label}]")
