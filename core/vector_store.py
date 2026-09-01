import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def build_vector_store(transcript: str, persist_directory: str = CHROMA_DIR) -> Chroma:
    if not transcript or not transcript.strip():
        raise ValueError("Cannot build vector store from empty transcript.")

    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    print(f"Vector store built successfully with {len(docs)} document chunks.")
    return vector_store


def load_vector_store(persist_directory: str = CHROMA_DIR) -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    sample_transcript = (
        "In our meeting we discussed the architecture of the AI Meeting Assistant. "
        "We selected OpenAI Whisper for local transcription and Sarvam AI for Indic translation. "
        "For the vector database, we chose ChromaDB with all-MiniLM-L6-v2 embeddings. "
        "Mistral Small model handles summarization and structured extraction of action items."
    )

    print("=== Testing Vector Store Creation ===")
    vs = build_vector_store(sample_transcript)

    print("\n=== Testing Similarity Search Retriever ===")
    retriever = get_retriever(vs, k=2)
    query = "Which vector database and embeddings model were selected?"
    results = retriever.invoke(query)
    print(f"Query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\nMatch {i + 1} (Chunk #{doc.metadata.get('chunk_index')}):")
        print(doc.page_content)

