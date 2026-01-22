from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid

client = QdrantClient(":memory:")  # local for demo

COLLECTION = "deepfake_faces"

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

def store_embedding(embedding, label):
    client.upsert(
        collection_name=COLLECTION,
        points=[{
            "id": str(uuid.uuid4()),
            "vector": embedding.tolist(),
            "payload": {"label": label}
        }]
    )
