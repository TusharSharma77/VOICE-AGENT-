import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.rate_limiter import redis_rate_limit, retry_with_backoff
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

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


@retry_with_backoff(max_retries=5, initial_delay=2.0)
@redis_rate_limit(key="mistral_llm", max_requests=5, window_seconds=60)
def _invoke_llm(chain, data):
    return chain.invoke(data)


def split_transcript(transcript: str) -> list:
    if not transcript or not transcript.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No transcript provided to summarize."

    chunks = split_transcript(transcript)
    if not chunks:
        return "No content to summarize."

    llm = get_llm()

    # 1. Map step: Summarize each chunk
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    # If only one chunk, single-step summary is sufficient
    if len(chunks) == 1:
        single_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert meeting summarizer. Generate a professional meeting summary "
                    "in structured bullet points with key discussion points, decisions, and action items.",
                ),
                ("human", "{text}"),
            ]
        )
        single_chain = single_prompt | llm | StrOutputParser()
        return _invoke_llm(single_chain, {"text": chunks[0]})

    chunk_summaries = [_invoke_llm(map_chain, {"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    # 2. Reduce step: Combine partial summaries
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in structured bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return _invoke_llm(combined_chain, combined)


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Meeting Summary"

    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Based on the meeting transcript, generate a short professional meeting title "
                    "(max 8 words). Only return the title, nothing else.",
                ),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )

    return _invoke_llm(title_chain, transcript[:2000]).strip()


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    sample_transcript = (
        "Welcome everyone to today's AI architecture sync. In this meeting we compared Agent Development Kits (ADK) "
        "and Retrieval-Augmented Generation (RAG). ADK is great when you need agentic tool-calling workflows and "
        "autonomous multi-step execution. RAG is best when you have structured knowledge bases and need factual "
        "grounding. We decided to combine both for our meeting assistant pipeline. Action item for Alex: integrate "
        "Whisper for transcription. Action item for Sam: connect Mistral LLM for summarization."
    )

    print("--- Testing generate_title ---")
    title = generate_title(sample_transcript)
    print(f"Generated Title: {title}\n")

    print("--- Testing summarize ---")
    summary = summarize(sample_transcript)
    print(f"Generated Summary:\n{summary}")



