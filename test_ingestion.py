from rag import RAGEngine

rag = RAGEngine()

results = rag.search(

    "What is deep learning?",

    top_k=3,

)

print()

for idx, result in enumerate(results, start=1):

    print("=" * 60)

    print(f"Result {idx}")

    print()

    print("Score :", result["score"])

    print("Source :", result["source"])

    print("Type :", result["type"])

    print("Chunk :", result["chunk_id"])

    print()

    print(result["text"])

rag.client.close()