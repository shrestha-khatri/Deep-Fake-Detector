from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

info = client.get_collection("deepfake_faces")
print(info)
