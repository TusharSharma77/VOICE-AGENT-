import os
import sys
from pathlib import Path

# Ensure project root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.rate_limiter import retry_with_backoff, redis_rate_limit
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

# Load environment variables from .env
load_dotenv()


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
        max_retries=4,
    )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _create_rag_chain(retriever):
    """Helper to assemble the RAG LCEL chain given a retriever."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_rag_chain(transcript: str, k: int = 4):
    """Build vector store from transcript and return a ready-to-query RAG chain."""
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=k)
    return _create_rag_chain(retriever)


def load_rag_chain(k: int = 4):
    """Load existing persisted vector store and return a ready-to-query RAG chain."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=k)
    return _create_rag_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    if not question or not question.strip():
        return "Please provide a valid non-empty question."
    print(f"Question : {question}")

    @retry_with_backoff(max_retries=3, initial_delay=2.0)
    @redis_rate_limit(key="mistral_llm", max_requests=10, window_seconds=60)
    def _invoke_chain():
        return rag_chain.invoke(question)

    try:
        answer = _invoke_chain()
    except Exception as e:
        answer = f"Error querying RAG engine: {e}"
    print(f"Answer   : {answer}")
    return answer


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Load existing vector database if available, else build from a default transcript
    if os.path.exists("vector_db"):
        print("Loading existing vector store from 'vector_db/'...")
        chain = load_rag_chain(k=4)
    else:
        sample_transcript = (
            "Project Phoenix Status Update Meeting. "
            "Lead engineer Maria announced that Phase 1 deployment is scheduled for October 15th. "
            "The team selected PostgreSQL 16 over MySQL due to JSONB performance requirements. "
            "DevOps engineer David confirmed that Kubernetes cluster migration is 80% complete. "
            "Budget approval for the Datadog APM tool is still pending VP approval."
        )
        print("No existing vector_db found. Building from sample transcript...")
        chain = build_rag_chain(sample_transcript, k=2)

    # Option A: If question provided as command-line argument, answer it directly
    if len(sys.argv) > 1:
        user_question = " ".join(sys.argv[1:])
        ask_question(chain, user_question)
    else:
        # Option B: Interactive chat loop (type your own questions!)
        print("\n" + "=" * 50)
        print("  Meeting Assistant RAG Chat (Interactive Mode)")
        print("  Type your question below (or type 'exit' to quit)")
        print("=" * 50 + "\n")

        while True:
            try:
                user_question = input("Ask a question: ").strip()
                if not user_question:
                    continue
                if user_question.lower() in ("exit", "quit", "q"):
                    print("Exiting RAG chat.")
                    break
                ask_question(chain, user_question)
                print("-" * 50)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting RAG chat.")
                break