from qdrant_client import QdrantClient

# Connect to local Qdrant
client = QdrantClient(host="localhost", port=6333)

COLLECTION = "deepfake_faces"

def predict(embedding, k=5):
    # Query similar embeddings from Qdrant
    response = client.query_points(
        collection_name=COLLECTION,
        query=embedding.tolist(),
        limit=k
    )

    # Extract labels from retrieved points
    labels = []
    for point in response.points:
        if point.payload and "label" in point.payload:
            labels.append(point.payload["label"])

    # Safety check
    if len(labels) == 0:
        return "Unknown", 0.0

    fake_count = labels.count("fake")
    real_count = labels.count("real")

    total = fake_count + real_count
    fake_ratio = fake_count / total

    if fake_ratio > 0.5:
        return "Deepfake", fake_ratio
    else:
        return "Real", 1 - fake_ratio
