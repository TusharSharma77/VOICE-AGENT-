# AI Meeting Assistant & Video Agent
<img width="1900" height="900" alt="image" src="https://github.com/user-attachments/assets/37d5f432-0056-4129-a97c-74ce1660bbda" />


An end-to-end multi-modal intelligence platform that ingests YouTube videos or local audio/video files, transcribes speech with local Whisper or Sarvam AI, synthesizes executive notes with Mistral AI using Map-Reduce, indexes content into ChromaDB, and enables interactive question-answering (RAG) through a Redis-protected API and a minimalist React dashboard.

---

## 1. System Architecture & Workflow Graph

The diagram below illustrates the end-to-end flow of data from ingestion to interactive Q&A:

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion Layer"]
        A1[YouTube URL] --> B1[yt-dlp Audio Extraction]
        A2[Local MP3 / WAV / MP4] --> B2[pydub Audio Normalization]
        B1 --> B2
        B2 --> C1[16kHz Mono WAV Normalization]
        C1 --> C2[Audio Chunking / Slicing]
    end

    subgraph TRANSCRIPTION["2. Speech-to-Text Engine"]
        C2 --> D{Language Choice}
        D -- "English" --> E1[OpenAI Whisper Local CPU/CUDA]
        D -- "Hinglish / Hindi" --> E2[Sarvam AI saaras:v2.5 STT-Translate API]
        E1 --> F[Compiled Transcript]
        E2 --> F
    end

    subgraph INTELLIGENCE["3. Intelligence & Extraction Layer"]
        F --> G1[Title Generator (Mistral AI)]
        F --> G2[Map-Reduce Summarizer (Mistral AI)]
        F --> G3[Action Items Extractor (Task, Owner, Deadline)]
        F --> G4[Key Decisions Extractor]
        F --> G5[Open Questions Extractor]
    end

    subgraph STORAGE["4. Storage & Indexing"]
        F --> H1[Recursive Character Text Splitter]
        H1 --> H2[HuggingFace all-MiniLM-L6-v2 Embeddings]
        H2 --> H3[(ChromaDB Vector Store)]
        G1 & G2 & G3 & G4 & G5 --> H4[(outputs/meeting_notes.md & history.json)]
    end

    subgraph RESILIENCE["5. Distributed Resilience"]
        R1[(Redis in Docker :6379)] <--> R2[Sliding-Window Rate Limiter]
        R2 <--> R3[Exponential Backoff Retries 429/503]
        R3 -. Protects .-> G1 & G2 & G3 & G4 & G5 & E2
    end

    subgraph RAG["6. Contextual Retrieval (RAG Engine)"]
        Q[User Query] --> K1[Similarity Search Retriever k=4]
        H3 --> K1
        K1 --> K2[Format Documents Lambda]
        K2 & Q --> K3[Strict Context Prompt Template]
        K3 --> K4[ChatMistralAI]
        K4 --> K5[Grounded Answer / Fallback Guard]
    end

    subgraph UI["7. Presentation Layer"]
        M1[CLI Interface: main.py] --> INGESTION
        M2[FastAPI Server: server.py :8000] --> INGESTION & RAG
        M3[React Monochrome Dashboard :5173] <--> M2
    end
```

---

## 2. Repository File Structure

```text
Video Agent/
│
├── core/                               # Core AI and processing modules
│   ├── __init__.py                     # Package marker
│   ├── transcriber.py                  # Whisper & Sarvam AI STT pipeline
│   ├── summarize.py                    # Map-Reduce summarization & title generation
│   ├── extractor.py                    # Action items, key decisions, open questions
│   ├── vector_store.py                 # ChromaDB embedding & retrieval indexer
│   └── rag_engine.py                   # Contextual RAG question-answering engine
│
├── utils/                              # Shared utility services
│   ├── __init__.py                     # Package marker
│   ├── audio_processor.py              # Download, convert to 16kHz WAV, chunking
│   └── rate_limiter.py                 # Redis sliding-window limiter & retry backoff
│
├── frontend/                           # React 19 + Vite frontend application
│   ├── src/
│   │   ├── App.jsx                     # 2-Panel zero-scroll dashboard component
│   │   ├── index.css                   # Patterned dark canvas & monochrome styles
│   │   └── main.jsx                    # React DOM entrypoint
│   ├── index.html                      # HTML template
│   ├── package.json                    # Node dependencies
│   └── vite.config.js                  # Vite configuration
│
├── downloades/                         # Staging directory for audio downloads & chunks
├── outputs/                            # Persisted transcripts and structured reports
│   ├── transcript.txt                  # Raw full meeting transcript
│   ├── meeting_notes.md                # Markdown report with summary & extractions
│   └── history.json                    # Document metadata store for API
│
├── vector_db/                          # Persisted ChromaDB vector database files
├── main.py                             # Unified CLI end-to-end pipeline runner
├── server.py                           # FastAPI REST API backend (port 8000)
├── Requirements.txt                    # Python package dependencies
├── .env                                # Environment variables & API credentials
└── README.md                           # Comprehensive documentation (this file)
```

---

## 3. Deep Dive: Module-by-Module Explanation

### 1. `utils/audio_processor.py` (Audio Pipeline)
* **`download_youtube_audio(url: str) -> str`**:
  * Uses `yt-dlp` to extract the best available audio stream from any YouTube video.
  * Leverages FFmpeg via `FFmpegExtractAudio` post-processor to encode audio directly to WAV at 192kbps.
* **`convert_to_wav(input_path: str) -> str`**:
  * Uses `pydub.AudioSegment` to normalize any audio/video container (`.mp4`, `.mp3`, `.m4a`, `.webm`, `.aac`) into standard mono 16kHz WAV format (16-bit PCM), matching the optimal input requirements for speech models.
* **`chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]`**:
  * Slices lengthy recordings into 10-minute segments with automatic naming (`_chunk_0.wav`, `_chunk_1.wav`) to avoid memory bottlenecks and stay within sync API upload constraints.
* **`process_input(source: str) -> list[str]`**:
  * Unified router: automatically sanitizes inputs by stripping enclosing quotes and whitespace.
  * Detects whether the input is a YouTube URL (`youtu.be` / `youtube.com`) or a local file, converts, chunks, and returns the list of ready-to-transcribe audio slices.

---

### 2. `utils/rate_limiter.py` (Redis Distributed Limiter & Resilience)
* **`RedisRateLimiter(key: str, max_requests: int, window_seconds: int)`**:
  * Implements a **Sliding Window Log** algorithm using Redis Sorted Sets (`ZADD`, `ZREMRANGEBYSCORE`, `ZCARD`).
  * Enforces exact rate limits across multiple workers/processes (e.g. 5 requests per 60s for Mistral, 20 requests per 60s for Sarvam).
  * If the quota is exceeded, dynamically calculates the exact sleep duration until the oldest request leaves the sliding window.
  * Gracefully passes through if Redis Docker is unreachable.
* **`@redis_rate_limit(key, max_requests, window_seconds)`**:
  * Function decorator to enforce Redis sliding-window spacing on API endpoints.
* **`@retry_with_backoff(max_retries=5, initial_delay=3.0, backoff_factor=2.0)`**:
  * Intercepts `429 Too Many Requests` and `503 Service Unavailable` server overloads.
  * Logs a clear retry warning and pauses with exponential backoff (`3.0s → 6.0s → 12.0s...`) instead of terminating the pipeline.

---

### 3. `core/transcriber.py` (Speech-to-Text Orchestrator)
* **`load_model()`**:
  * Singleton loader for OpenAI Whisper (`whisper.load_model()`). Reads `WHISPER_MODEL` (e.g. `small`, `base`) from `.env`.
* **`transcribe_chunk_whisper(chunk_path: str) -> str`**:
  * Transcribes English audio locally using Whisper.
  * Automatically detects `torch.cuda.is_available()`. Sets `fp16=True` on CUDA GPUs, and silently falls back to `fp16=False` on CPUs to eliminate PyTorch FP16 warnings.
* **`transcribe_chunk_sarvam(chunk_path: str) -> str`**:
  * Transcribes and translates Hindi/Hinglish audio into English using Sarvam AI's `saaras:v2.5` model.
  * Slices 10-minute chunks into ≤25-second micro-pieces to respect Sarvam's 30-second API limit.
  * Protected by `@redis_rate_limit(key="sarvam_stt")` and `@retry_with_backoff`.
* **`transcribe_all(chunks: list[str], language: str) -> str`**:
  * Iterates through all chunks, transcribes each sequentially with progress logging, and concatenates them into one complete transcript.

---

### 4. `core/summarize.py` (Map-Reduce Summarizer & Title Generator)
* **`get_llm()`**:
  * Initializes `ChatMistralAI` (`mistral-small-latest`) with `temperature=0.3` and `max_retries=4`.
* **`generate_title(transcript: str) -> str`**:
  * Reads the meeting introduction and generates a concise, professional title (≤ 8 words).
* **`split_transcript(transcript: str) -> list[str]`**:
  * Uses `RecursiveCharacterTextSplitter` with `chunk_size=3000` and `chunk_overlap=200` to partition transcripts that exceed LLM context windows.
* **`summarize(transcript: str) -> str`**:
  * **Single-step**: If the transcript fits in one chunk, directly produces structured notes.
  * **Map-Reduce**: If multi-chunk, runs a Map step over each chunk to generate partial summaries, then executes a Reduce step combining partial notes into an executive summary with key discussion points, decisions, and action items.
  * Wrapped in `@redis_rate_limit(key="mistral_llm")` and `@retry_with_backoff`.

---

### 5. `core/extractor.py` (Action Items, Decisions & Questions)
* **`build_chain(system_prompt: str)`**:
  * Assembles an LCEL pipeline: `ChatPromptTemplate | ChatMistralAI(temp=0.2) | StrOutputParser()`.
* **`extract_action_items(transcript: str) -> str`**:
  * Extracts structured tasks, responsible owners, and deadlines.
* **`extract_key_decisions(transcript: str) -> str`**:
  * Identifies all concrete agreements and architectural/business decisions made.
* **`extract_questions(transcript: str) -> str`**:
  * Surfaces unresolved questions or pending discussion points requiring follow-up.
* **`extract_all(transcript: str) -> dict`**:
  * Executes all three extractors, returning a unified dictionary. Each call is rate-limited via Redis.

---

### 6. `core/vector_store.py` (ChromaDB Indexer & Embeddings)
* **`get_embeddings()`**:
  * Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` (running locally on CPU/GPU without external API rate limits).
* **`build_vector_store(transcript: str, persist_directory: str = "vector_db")`**:
  * Splits the transcript into semantic chunks (`chunk_size=500, chunk_overlap=50`).
  * Computes vector embeddings and saves to the local `vector_db/` directory via ChromaDB.
* **`load_vector_store(persist_directory: str = "vector_db")`**:
  * Loads the existing on-disk ChromaDB vector store for instant querying without re-indexing.
* **`get_retriever(vector_store, k: int = 4)`**:
  * Returns a similarity search retriever configured to fetch the top-k most relevant chunks.

---

### 7. `core/rag_engine.py` (Contextual Q&A Engine)
* **`_create_rag_chain(retriever)`**:
  * Assembles the full LCEL RAG chain:
    ```python
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    ```
  * Strict anti-hallucination prompt: instruct the model to answer **only** using provided context, falling back to *"I could not find this information in the meeting transcript."* if missing.
* **`ask_question(rag_chain, question: str) -> str`**:
  * Queries the chain with `@retry_with_backoff` and `@redis_rate_limit`.
* **Interactive CLI Runner (`__main__`)**:
  * Supports both one-shot CLI arguments (`python core/rag_engine.py "my question"`) and an interactive REPL terminal loop.

---

### 8. `main.py` (Unified CLI Entrypoint)
* Coordinates the full 5-stage pipeline from terminal:
  1. `[1/5]` Process audio (download, WAV conversion, chunking)
  2. `[2/5]` Transcribe chunks (Whisper or Sarvam AI)
  3. `[3/5]` Generate title and executive summary
  4. `[4/5]` Extract action items, decisions, and questions
  5. `[5/5]` Build vector store & launch interactive Q&A REPL
* Automatically persists results into `outputs/transcript.txt` and `outputs/meeting_notes.md`.

---

### 9. `server.py` (FastAPI Backend)
* High-performance ASGI web server exposing REST endpoints on port `8000`:
  * `GET /api/status`: Health check for API and Redis connection.
  * `GET /api/documents`: Fetches list of all processed meetings/documents from `outputs/history.json`.
  * `POST /api/process`: Ingests a YouTube URL or file path, executes the pipeline, and returns notes.
  * `POST /api/upload`: Multipart file upload for local audio/video files.
  * `POST /api/ask`: Queries the indexed ChromaDB RAG chain with user questions.
* Configured with `CORSMiddleware` to allow communication with the React frontend.

---

### 10. `frontend/` (React 19 + Vite Monochrome Dashboard)
* **Zero-Scroll Canvas**: Styled with a dark viewport (`#09090b`), subtle dot matrix grid pattern (`28px` spacing), and ambient spotlight. No outer scrollbars.
* **Centered Workspace Window**: macOS window header (red, yellow, green dots), clean borders, and drop shadow.
* **2-Panel Split Interior**:
  * **Left Panel**: Media acquisition form (YouTube URL or path input, language selector, Process button) + tabbed document viewer (**Summary**, **Action Items**, **Key Decisions**, **Questions**, **Transcript**).
  * **Right Panel**: AI Assistant chat stream with dark user speech bubbles, light grounded assistant response cards, and bottom input field with send trigger.
* Strictly zero emojis; clean monochrome high-contrast aesthetic.

---

## 4. Setup and Installation

### Prerequisites
* **Python**: 3.10, 3.11, or 3.12
* **Node.js**: v18+ (or Bun)
* **FFmpeg**: Required by `yt-dlp` and `pydub` (installed via `winget install Gyan.FFmpeg` or system package manager)
* **Docker**: Required for Redis rate limiting (`docker run -d -p 6379:6379 redis:alpine`)

---

### Step 1: Clone and Setup Python Environment

```powershell
# Navigate to project directory
cd "Video Agent"

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
uv pip install -r Requirements.txt --link-mode copy
# or: pip install -r Requirements.txt
```

---

### Step 2: Configure Environment Variables (`.env`)

Create or update [.env](file:///c:/Users/HP/OneDrive/Desktop/Video%20Agent/.env) in the root directory:

```ini
# Mistral AI (Summarization, Extraction, RAG)
MISTRAL_API_KEY="your_mistral_api_key_here"

# Whisper Local Model ("small", "base", or "tiny")
WHISPER_MODEL="small"

# Sarvam AI (Hinglish/Hindi STT & Translation)
SARVAM_API_KEY="your_sarvam_api_key_here"
SARVAM_STT_MODEL="saaras:v2.5"

# Redis Docker Rate Limiter
REDIS_URL="redis://localhost:6379/0"
```

---

### Step 3: Start Redis in Docker

```powershell
# If starting for the first time:
docker run -d --name redis -p 6379:6379 redis:alpine

# Or start existing container:
docker start redis
```

---

### Step 4: Setup Frontend

```powershell
cd frontend
bun install
# or: npm install
cd ..
```

---

## 5. How to Run

### Mode A: Full Web Dashboard (Recommended)

Start the backend and frontend in two separate terminal windows:

#### Terminal 1 — Backend Server:
```powershell
.\.venv\Scripts\Activate.ps1
python server.py
```
*API running at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`)*

#### Terminal 2 — React Dashboard:
```powershell
cd frontend
bun run dev
# or: npm run dev
```
*Open your browser at **`http://localhost:5173`**.*

---

### Mode B: Command-Line Pipeline (`main.py`)

Run the entire pipeline directly in your terminal without opening a browser:

```powershell
# Process English video
python main.py "https://youtu.be/ZELPNFXJ4_o?si=DIuhSpciet1zKW0o"

# Process Hinglish video
python main.py "https://youtu.be/..." hinglish

# Process local audio file
python main.py "path/to/meeting.mp3"
```

---

### Mode C: Standalone Module Testing

Every module has its own `if __name__ == "__main__":` test block:

| Test Target | Command |
| :--- | :--- |
| **Audio Processing** | `python utils/audio_processor.py "<youtube_url>"` |
| **Redis Limiter** | `python utils/rate_limiter.py` |
| **Transcription** | `python core/transcriber.py "<chunk.wav>" english` |
| **Summarization** | `python core/summarize.py` |
| **Extractions** | `python core/extractor.py` |
| **Vector Store** | `python core/vector_store.py` |
| **RAG Interactive** | `python core/rag_engine.py` *(or pass `"your question"`)* |

---

## 6. Production Deployment Guide

### Option 1: Docker Compose (All-in-One — Recommended for Cloud VPS)
Works on any Linux server (AWS EC2, DigitalOcean Droplet, Hetzner, GCP Compute Engine):

1. **Clone the repo onto your server**:
   ```bash
   git clone <your-repo-url>
   cd "Video Agent"
   ```
2. **Create `.env` file** with your API keys:
   ```bash
   nano .env
   # Add MISTRAL_API_KEY, SARVAM_API_KEY, WHISPER_MODEL, etc.
   ```
3. **Launch the entire stack**:
   ```bash
   docker compose up -d --build
   ```
   * Frontend will be live on port `80` (or your domain).
   * Backend API runs on port `8000`.
   * Redis runs internally on port `6379`.

---

### Option 2: Render.com / Railway.app (PaaS Deployment)

#### Backend (FastAPI + FFmpeg + Whisper):
1. Create a new **Web Service** from your GitHub repo.
2. Select **Docker** as the environment (it will automatically use [Dockerfile](file:///c:/Users/HP/OneDrive/Desktop/Video%20Agent/Dockerfile)).
3. Add Environment Variables:
   * `MISTRAL_API_KEY`
   * `SARVAM_API_KEY`
   * `WHISPER_MODEL=base` *(use `base` or `small` for CPU tiers)*
   * `REDIS_URL` *(link to a Redis instance)*
4. Attach a **Persistent Disk** mounted at `/app/vector_db` and `/app/outputs` to preserve embeddings across redeploys.

#### Redis:
* Add a Redis service on Render/Railway with 1 click, and set `REDIS_URL` in the backend service.

#### Frontend (Vercel / Netlify / Render Static):
1. Deploy the `frontend/` folder to Vercel or Netlify.
2. Set Environment Variable:
   * `VITE_API_BASE=https://your-backend-api.onrender.com`
3. Build command: `bun run build` (or `npm run build`), Output directory: `dist`.

---

## 7. Resilience & Production Features

1. **Automatic Quote Stripping**: `source.strip().strip('"').strip("'")` handles PowerShell and command-line quoting without throwing `OSError: [Errno 22]`.
2. **CPU FP16 Warning Elimination**: Whisper automatically selects `fp16=torch.cuda.is_available()`, avoiding CPU half-precision warnings.
3. **Sliding-Window Distributed Quotas**: Redis coordinates API request budgets across all processes.
4. **Transient 503 Self-Healing**: Catches temporary upstream provider overloads, logs warnings, and applies exponential backoff retries (`3s → 6s → 12s`) instead of failing.
5. **Zero-Hallucination Guard**: The RAG prompt strictly restricts answers to the meeting transcript and provides clear fallback statements when information is absent.
