import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTORE_SIZE = 128
COLLECTION_NAME = "test_collection"

client = QdrantClient(path="../storage")

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTORE_SIZE, distance=Distance.COSINE),
    )

vectors = [
    PointStruct(id=1, vector=np.random.rand(VECTORE_SIZE).tolist(), payload={"name": "vector1"}),
    PointStruct(id=2, vector=np.random.rand(VECTORE_SIZE).tolist(), payload={"name": "vector2"}),
    PointStruct(id=3, vector=np.random.rand(VECTORE_SIZE).tolist(), payload={"name": "vector3"}),
]
client.upsert(collection_name="test_collection", points=vectors)

search_result = client.search(
    collection_name=COLLECTION_NAME, 
    query_vector=np.random.rand(VECTORE_SIZE).tolist(), 
    limit=1
)

# Ergebnisse anzeigen
for hit in search_result:
    print(f"ID: {hit.id}, Score: {hit.score}, Payload: {hit.payload}")
