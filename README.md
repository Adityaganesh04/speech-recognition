# Meeting Intelligence System — Stage 1: Foundation

A modular, multi-speaker meeting transcription and semantic search system built on strict layered architecture principles.

---

## Architecture

```
Audio Input
    ↓
[Layer 1] Speech Processing    → SpeechSegment[]     (WhisperX ASR + Diarization)
    ↓
[Layer 2] Data Structuring     → Chunk[]              (Chunking + Metadata enrichment)
    ↓
[Layer 3] Intelligence         → IntelligenceOutput   (Summarization + Entity extraction)
    ↓
[Layer 4] Storage              → ChromaDB + SQLite    (Vector store + Relational store)
    ↓
[Layer 5] Query & Retrieval    → RetrievedChunk[]     (Semantic search)
    ↓
[Layer 6] Output Generation    → OutputResponse       (Grounded answer + Sources)
```

---

## Project Structure

```
new final/
├── main.py                          # Pipeline orchestrator + CLI
├── config.py                        # Centralized configuration (all settings)
├── requirements.txt                 # All dependencies
│
├── contracts/
│   └── schemas.py                   # Versioned Pydantic data contracts
│
├── layers/
│   ├── speech_processing/
│   │   └── processor.py             # Layer 1: WhisperX ASR + diarization
│   ├── data_structuring/
│   │   └── structurer.py            # Layer 2: Chunking + metadata
│   ├── intelligence/
│   │   └── extractor.py             # Layer 3: Summarization + NER
│   ├── storage/
│   │   └── store.py                 # Layer 4: ChromaDB + SQLite
│   ├── query_retrieval/
│   │   └── retriever.py             # Layer 5: Semantic search
│   └── output/
│       └── generator.py             # Layer 6: Answer + source attribution
│
├── utils/
│   └── logger.py                    # Structured layer-aware logging
│
└── data/
    ├── audio/                       # Drop audio files here
    ├── meetings/                    # JSON artifacts per meeting
    └── db/
        ├── chroma/                  # ChromaDB vector store
        └── meetings.db              # SQLite metadata store
```

---

## Setup

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Install spaCy English model

```bash
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

### 3. (Optional) Enable speaker diarization

Diarization requires a free HuggingFace token:
1. Create account at https://huggingface.co
2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization
3. Set environment variable:

```powershell
# Windows
$env:HF_TOKEN = "hf_your_token_here"

# Linux / Mac
export HF_TOKEN="hf_your_token_here"
```

Without `HF_TOKEN`, all speakers will be labeled `SPEAKER_00` — transcription still works fully.

### 4. (Optional) GPU acceleration

```powershell
$env:DEVICE = "cuda"
$env:COMPUTE_TYPE = "float16"
```

---

## Usage

### Ingest a meeting audio file

```bash
python main.py ingest path/to/meeting.mp3
python main.py ingest path/to/meeting.wav --id my_meeting_001
```

Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.mp4`

### Query across all meetings

```bash
python main.py query "What was decided about the project deadline?"
python main.py query "Who mentioned the budget concerns?"
python main.py query "What action items were assigned to the team?"
```

### Query within a specific meeting

```bash
python main.py query "What did the manager say about delivery?" --meeting mtg_abc12345
```

### List all stored meetings

```bash
python main.py list
```

### View meeting summary & entities

```bash
python main.py summary mtg_abc12345
```

---

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable              | Default                    | Description                        |
|-----------------------|----------------------------|------------------------------------|
| `WHISPER_MODEL`       | `base`                     | ASR model: tiny/base/small/medium  |
| `DEVICE`              | `cpu`                      | `cpu` or `cuda`                    |
| `COMPUTE_TYPE`        | `int8`                     | `int8` (CPU) or `float16` (GPU)    |
| `HF_TOKEN`            | *(empty)*                  | HuggingFace token for diarization  |
| `MAX_CHUNK_TOKENS`    | `200`                      | Max words per chunk                |
| `MIN_CHUNK_TOKENS`    | `20`                       | Min words per chunk                |
| `EMBEDDING_MODEL`     | `all-MiniLM-L6-v2`         | Sentence embedding model           |
| `DEFAULT_TOP_K`       | `5`                        | Number of search results returned  |
| `SIMILARITY_THRESHOLD`| `0.3`                      | Minimum cosine similarity to return|

---

## Upgrade Path (Stage 2)

Each layer is independently upgradeable:

| Layer | Stage 1 | Stage 2 Upgrade |
|-------|---------|-----------------|
| Speech | WhisperX base | Whisper large-v3, domain-specific |
| Structuring | Token-bounded chunks | Semantic boundary detection |
| Intelligence | BART summarization | GPT-4 / Gemini / Llama3 |
| Storage | ChromaDB local | Pinecone / Weaviate / pgvector |
| Retrieval | Dense (sentence-transformers) | Hybrid BM25 + dense + reranker |
| Output | Template-based | LLM-grounded (swap `_generate()`) |

---

## Adding LLM-backed answers (Stage 2 preview)

In `layers/output/generator.py`, replace `_generate()` with:

```python
def _generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
    prompt = self._build_prompt(query, chunks)
    # Ollama (local):
    import ollama
    response = ollama.chat(model="llama3", messages=[{"role":"user","content":prompt}])
    return response["message"]["content"]
    # OpenAI:
    # from openai import OpenAI
    # client = OpenAI()
    # return client.chat.completions.create(model="gpt-4o", messages=[...]).choices[0].message.content
```

No other code changes needed — all contracts remain the same.
