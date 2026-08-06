from rag import RAGEngine

rag = RAGEngine()

answer, results = rag.generate_answer(

    "What is Deep Learning?"

)

print()

print(answer)

print()

print("Sources Used")

for r in results:

    print(

        r.payload["source_name"],

        "Chunk",

        r.payload["chunk_id"],

    )

rag.client.close()