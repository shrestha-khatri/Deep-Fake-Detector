from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="deepfake_faces",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)

print("Qdrant collection created")
