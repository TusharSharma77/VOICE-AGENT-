# Action items, key decisions, questions extractor
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
        temperature=0.2,
        max_retries=4,
    )


@retry_with_backoff(max_retries=5, initial_delay=2.0)
@redis_rate_limit(key="mistral_llm", max_requests=5, window_seconds=60)
def _invoke_chain(chain, transcript: str) -> str:
    return chain.invoke(transcript)


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No action items found (empty transcript)."
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    return _invoke_chain(chain, transcript)


def extract_key_decisions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No key decisions found (empty transcript)."
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return _invoke_chain(chain, transcript)


def extract_questions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No open questions found (empty transcript)."
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return _invoke_chain(chain, transcript)


def extract_all(transcript: str) -> dict:
    """Extract action items, key decisions, and questions in one dictionary (Redis rate-limited)."""
    return {
        "action_items": extract_action_items(transcript),
        "key_decisions": extract_key_decisions(transcript),
        "questions": extract_questions(transcript),
    }


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    sample_transcript = (
        "Welcome everyone to today's product sync. In this meeting we decided to adopt FastAPI for the backend "
        "and PostgreSQL for the database. Action item for Priya: set up the FastAPI repo by Friday. "
        "Action item for John: write database migrations by next Tuesday. "
        "Open question from Rahul: Do we need Redis for rate limiting in the first MVP release?"
    )

    print("=== Extracting Action Items ===")
    print(extract_action_items(sample_transcript))
    print("\n=== Extracting Key Decisions ===")
    print(extract_key_decisions(sample_transcript))
    print("\n=== Extracting Open Questions ===")
    print(extract_questions(sample_transcript))