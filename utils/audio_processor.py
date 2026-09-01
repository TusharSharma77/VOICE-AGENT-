import  yt_dlp
from pydub import AudioSegment
import os
DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        filename = base + ".wav"
    return filename

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split WAV file into smaller chunks of specified duration."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []
    base_name = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base_name}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    """Download or convert input source and split it into chunks."""
    if not source:
        raise ValueError("Source input cannot be empty.")

    # Strip whitespace and any surrounding single or double quotes
    source = source.strip().strip('"').strip("'").strip()

    is_url = (
        source.startswith("http://")
        or source.startswith("https://")
        or "youtube.com" in source.lower()
        or "youtu.be" in source.lower()
    )

    if is_url:
        print("Detected YouTube URL. Downloading audio...")
        raw_audio = download_youtube_audio(source)
        print("Converting downloaded audio to mono 16kHz WAV...")
        wav_path = convert_to_wav(raw_audio)
    else:
        print(f"Detected local file: {source}")
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local file not found at path: {source}")
        print("Converting local file to mono 16kHz WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

if __name__ == "__main__":
    import sys
    # Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows console
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        
    if len(sys.argv) < 2:
        print("Usage: python utils/audio_processor.py <youtube_url_or_local_file_path>")
        sys.exit(1)
        
    source = sys.argv[1]
    chunks = process_input(source)
    print("Generated chunks:")
    for chunk in chunks:
        print(f"  - {chunk}")