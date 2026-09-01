import os
import sys
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english"):
    """
    Run the end-to-end meeting assistant pipeline:
    1. Audio download / conversion & chunking
    2. Transcription (Whisper or Sarvam AI)
    3. Title generation & Map-Reduce summarization (Mistral AI)
    4. Action items, decisions & questions extraction (Mistral AI)
    5. Vector store indexing & Interactive RAG Q&A (ChromaDB + Mistral)
    """
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("       AI MEETING ASSISTANT & VIDEO AGENT PIPELINE")
    print("=" * 60)
    print(f"Source   : {source}")
    print(f"Language : {language}\n")

    # Step 1: Audio Processing
    print("[1/5] Processing Audio...")
    chunks = process_input(source)
    if not chunks:
        print("Error: No audio chunks created.")
        return

    # Step 2: Transcription
    print(f"\n[2/5] Transcribing {len(chunks)} chunk(s) with language '{language}'...")
    transcript = transcribe_all(chunks, language=language)
    print("\n--- Transcript Preview ---")
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))

    # Save transcript to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print("Full transcript saved to 'outputs/transcript.txt'")

    # Step 3: Title and Summary
    print("\n[3/5] Generating Meeting Title and Summary...")
    title = generate_title(transcript)
    summary = summarize(transcript)

    print(f"\nMeeting Title: {title}")
    print("\n--- Executive Summary ---")
    print(summary)

    # Step 4: Key Insights Extraction
    print("\n[4/5] Extracting Action Items, Decisions, and Questions...")
    insights = extract_all(transcript)

    print("\n--- Action Items ---")
    print(insights["action_items"])

    print("\n--- Key Decisions ---")
    print(insights["key_decisions"])

    print("\n--- Open Questions ---")
    print(insights["questions"])

    # Save full report
    with open("outputs/meeting_notes.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("## Executive Summary\n" + summary + "\n\n")
        f.write("## Action Items\n" + insights["action_items"] + "\n\n")
        f.write("## Key Decisions\n" + insights["key_decisions"] + "\n\n")
        f.write("## Open Questions\n" + insights["questions"] + "\n")
    print("\nFull meeting notes saved to 'outputs/meeting_notes.md'")

    # Step 5: Interactive RAG Q&A
    print("\n[5/5] Building Vector Store for Interactive Q&A...")
    rag_chain = build_rag_chain(transcript)

    print("\n" + "=" * 60)
    print("  Interactive RAG Q&A Ready!")
    print("  Ask any question about this meeting (type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting meeting assistant. Have a great day!")
                break
            ask_question(rag_chain, query)
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting meeting assistant.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_input = sys.argv[1].strip().strip('"').strip("'")
        lang_input = (sys.argv[2] if len(sys.argv) > 2 else "english").strip().strip('"').strip("'")
    else:
        print("Enter a YouTube URL or a path to a local audio/video file:")
        source_input = input("Source: ").strip().strip('"').strip("'")
        if not source_input:
            print("No source provided. Exiting.")
            sys.exit(0)
        raw_lang = input("Language (english / hinglish) [default: english]: ").strip().strip('"').strip("'")
        lang_input = raw_lang or "english"

    run_pipeline(source_input, language=lang_input)
