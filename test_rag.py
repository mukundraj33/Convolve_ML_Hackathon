from rag import RAGEngine

rag = RAGEngine()

text = """
Artificial Intelligence is transforming
education through personalized learning.
""" * 40

chunks = rag.chunk_text(text)

print()

print("Number of Chunks :", len(chunks))

print()

print(chunks[0])

print()

print(chunks[-1])