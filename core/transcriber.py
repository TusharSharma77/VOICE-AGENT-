import os
import sys
from pathlib import Path
import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

# Ensure project root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.rate_limiter import retry_with_backoff, redis_rate_limit

# Load environment variables from .env file
load_dotenv()

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():
    global _model  

    if _model is None: 
        model_name = os.getenv("WHISPER_MODEL", WHISPER_MODEL)
        print(f"Loading Whisper model: {model_name} ...")
        _model = whisper.load_model(model_name) 
        print("Whisper model loaded.")
    return _model 


import gc
import torch


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    use_fp16 = torch.cuda.is_available()
    result = model.transcribe(chunk_path, task="transcribe", fp16=use_fp16)
    gc.collect()
    return result["text"].strip()


@retry_with_backoff(max_retries=4, initial_delay=2.0)
@redis_rate_limit(key="sarvam_stt", max_requests=20, window_seconds=60)
def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    api_key = os.getenv("SARVAM_API_KEY", SARVAM_API_KEY)
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": api_key}
    model_name = os.getenv("SARVAM_STT_MODEL", SARVAM_MODEL)

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": model_name, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    api_key = os.getenv("SARVAM_API_KEY", SARVAM_API_KEY)
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    base_name = os.path.splitext(chunk_path)[0]

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{base_name}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = "" 
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)  
        full_transcript += text + " "  

    print("Transcription complete.")
    return full_transcript.strip()


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python core/transcriber.py <chunk_path_or_audio_path> [english|hinglish]")
        sys.exit(1)

    input_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "english"

    if os.path.exists(input_file):
        result_text = transcribe_all([input_file], language=lang)
        print("\n--- Transcription Result ---")
        print(result_text)
    else:
        print(f"Error: File not found: {input_file}")  