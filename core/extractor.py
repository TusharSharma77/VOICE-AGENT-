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


import re

def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No action items found (empty transcript)."
    chain = build_chain(
        "You are an expert analyst. From the transcript (meeting, talk, or discussion), "
        "extract all actionable tasks and practical next steps.\n"
        "- If a business/team meeting: extract task, owner, and deadline.\n"
        "- If an educational/technical talk or presentation: extract actionable takeaways, "
        "practical implementation steps, and recommended best practices for viewers.\n"
        "Format as a numbered list with bold titles."
    )
    return _invoke_chain(chain, transcript)


def extract_key_decisions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No key decisions found (empty transcript)."
    chain = build_chain(
        "You are an expert analyst. From the transcript (meeting, talk, or discussion), "
        "extract all key decisions, core conclusions, and architectural principles established.\n"
        "Format as a numbered list with bold titles."
    )
    return _invoke_chain(chain, transcript)


def extract_questions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No open questions found (empty transcript)."
    chain = build_chain(
        "You are an expert analyst. From the transcript (meeting, talk, or discussion), "
        "extract all unresolved questions, pending follow-ups, or open challenges/considerations raised.\n"
        "Format as a numbered list with bold titles."
    )
    return _invoke_chain(chain, transcript)


def extract_all(transcript: str) -> dict:
    """
    Extract action items, key decisions, and questions using a unified prompt
    to minimize API calls and prevent rate limiting while generating rich insights.
    """
    if not transcript or not transcript.strip():
        return {
            "action_items": "No action items found (empty transcript).",
            "key_decisions": "No key decisions found (empty transcript).",
            "questions": "No open questions found (empty transcript).",
        }

    chain = build_chain(
        "You are an expert intelligence analyst. Analyze this transcript, which may be a meeting, "
        "technical presentation, tutorial, or discussion.\n\n"
        "Extract three comprehensive sections formatted in clean Markdown:\n\n"
        "## ACTION ITEMS & IMPLEMENTATION STEPS\n"
        "Extract actionable tasks and implementation steps. If a business meeting, identify task descriptions, "
        "responsible owners, and deadlines. If an educational or technical talk, extract actionable takeaways, "
        "practical steps, and recommended practices for practitioners.\n\n"
        "## KEY DECISIONS & CORE PRINCIPLES\n"
        "Extract key decisions, conclusions, agreements, or architectural principles established.\n\n"
        "## OPEN QUESTIONS & FUTURE CHALLENGES\n"
        "Extract unresolved questions, pending follow-ups, or open challenges raised.\n\n"
        "Format each section as a clean numbered list with bold titles."
    )

    try:
        raw_output = _invoke_chain(chain, transcript)
        action_match = re.search(r'##\s*\**\s*ACTION ITEMS[^\n]*\n(.*?)(?=\n##|\Z)', raw_output, flags=re.IGNORECASE | re.DOTALL)
        decision_match = re.search(r'##\s*\**\s*KEY DECISIONS[^\n]*\n(.*?)(?=\n##|\Z)', raw_output, flags=re.IGNORECASE | re.DOTALL)
        question_match = re.search(r'##\s*\**\s*OPEN QUESTIONS[^\n]*\n(.*?)(?=\n##|\Z)', raw_output, flags=re.IGNORECASE | re.DOTALL)

        actions = action_match.group(1).strip() if action_match else ""
        decisions = decision_match.group(1).strip() if decision_match else ""
        questions = question_match.group(1).strip() if question_match else ""

        # Fallback if markdown headers were not used
        if not actions and not decisions and not questions:
            return {
                "action_items": extract_action_items(transcript),
                "key_decisions": extract_key_decisions(transcript),
                "questions": extract_questions(transcript),
            }

        return {
            "action_items": actions or "No action items found.",
            "key_decisions": decisions or "No key decisions found.",
            "questions": questions or "No open questions found.",
        }
    except Exception:
        # Fallback to individual calls if unified parse fails
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