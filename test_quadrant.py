from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

import uuid

from config import (
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    QDRANT_PATH,
)

print("Loading Embedding Model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to Qdrant...")

client = QdrantClient(path=QDRANT_PATH)

collections = client.get_collections().collections

names = [c.name for c in collections]

if COLLECTION_NAME not in names:

    client.create_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            size=model.get_sentence_embedding_dimension(),

            distance=Distance.COSINE,

        ),
    )

print("Collection Ready")

documents = [

    "Machine learning is a subset of Artificial Intelligence.",

    "Neural networks are inspired by the human brain.",

    "Deep learning uses multiple hidden layers.",

]

points = []

for i, doc in enumerate(documents):

    embedding = model.encode(doc).tolist()

    points.append(

        PointStruct(

            id=str(uuid.uuid4()),

            vector=embedding,

            payload={

                "text": doc,

                "id": i,

            },
        )
    )

client.upsert(

    collection_name=COLLECTION_NAME,

    points=points,

)

print("Documents Stored!")

query = "Explain deep learning"

query_vector = model.encode(query).tolist()

results = client.query_points(

    collection_name=COLLECTION_NAME,

    query=query_vector,

    limit=3,

)

print("\nRetrieved Results\n")

for point in results.points:

    print("-----------------------------")

    print(point.payload["text"])

    print("Score:", point.score)