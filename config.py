"""
config.py — Centralized Configuration
Single source of truth for all layer settings.
Override via environment variables where applicable.
Architecture principle: Plug-and-play (swap models here, not in layers).

Fix #10: Added startup validation to prevent silent misconfiguration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Disable ChromaDB telemetry to prevent tracking timeouts
os.environ["CHROMA_SERVER_TELEMETRY"] = "False"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
MEETINGS_DIR = DATA_DIR / "meetings"
DB_DIR = DATA_DIR / "db"

# Ensure directories exist on import
for _dir in [AUDIO_DIR, MEETINGS_DIR, DB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Speech Processing
# ─────────────────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")   # tiny | base | small | medium | large
DEVICE = os.getenv("DEVICE", "cuda")                  # "cuda" for GPU, "cpu" for fallback
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")   # "float16" for GPU, "int8" for CPU

# HuggingFace token required for pyannote diarization
# Set via: set HF_TOKEN=hf_xxxx  (Windows) or export HF_TOKEN=hf_xxxx (Linux/Mac)
HF_TOKEN = os.getenv("HF_TOKEN", "")
DIARIZATION_ENABLED = bool(HF_TOKEN)

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Data Structuring
# ─────────────────────────────────────────────────────────────────────────────
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "200"))  # Max words per chunk
MIN_CHUNK_TOKENS = int(os.getenv("MIN_CHUNK_TOKENS", "20"))   # Min words (below = discarded)
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "30"))  # Word overlap between chunks

# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Intelligence
# ─────────────────────────────────────────────────────────────────────────────
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH", "150"))
MIN_SUMMARY_LENGTH = int(os.getenv("MIN_SUMMARY_LENGTH", "40"))
USE_SPACY = os.getenv("USE_SPACY", "true").lower() == "true"
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Storage
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "meeting_chunks")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(DB_DIR / "chroma"))
SQLITE_PATH = os.getenv("SQLITE_PATH", str(DB_DIR / "meetings.db"))

# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — Embeddings & Retrieval
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 — Output Gen / Layer 3 — Intent Intel 
# ─────────────────────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")

# ─────────────────────────────────────────────────────────────────────────────
# Feature Flags (Section 5.3: Controlled feature rollout)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = {
    "enable_summarization": os.getenv("FEATURE_SUMMARIZATION", "true").lower() == "true",
    "enable_entity_extraction": os.getenv("FEATURE_ENTITY_EXTRACTION", "true").lower() == "true",
    "enable_intent_extraction": os.getenv("FEATURE_INTENT_EXTRACTION", "true").lower() == "true",
    "enable_diarization": DIARIZATION_ENABLED,
    "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "token_bounded"),  # future: "semantic"
    "output_mode": os.getenv("OUTPUT_MODE", "llm"),                        # Upgraded to LLM
}


# ─────────────────────────────────────────────────────────────────────────────
# Startup Validation (fix #10)
# Catches misconfigurations before they cause cryptic errors downstream.
# ─────────────────────────────────────────────────────────────────────────────

def _validate_config():
    """Validate all config values on import. Fail fast with clear messages."""
    errors = []

    # Layer 1
    valid_models = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    if WHISPER_MODEL not in valid_models:
        errors.append(f"WHISPER_MODEL='{WHISPER_MODEL}' is not valid. Choose from: {valid_models}")
    if DEVICE not in {"cpu", "cuda"}:
        errors.append(f"DEVICE='{DEVICE}' is not valid. Use 'cpu' or 'cuda'.")
    if COMPUTE_TYPE not in {"int8", "float16", "float32"}:
        errors.append(f"COMPUTE_TYPE='{COMPUTE_TYPE}' is not valid. Use 'int8', 'float16', or 'float32'.")

    # Layer 2
    if MAX_CHUNK_TOKENS < 1:
        errors.append(f"MAX_CHUNK_TOKENS={MAX_CHUNK_TOKENS} must be >= 1 (would cause infinite loop).")
    if MIN_CHUNK_TOKENS < 0:
        errors.append(f"MIN_CHUNK_TOKENS={MIN_CHUNK_TOKENS} must be >= 0.")
    if MIN_CHUNK_TOKENS >= MAX_CHUNK_TOKENS:
        errors.append(
            f"MIN_CHUNK_TOKENS ({MIN_CHUNK_TOKENS}) must be < MAX_CHUNK_TOKENS ({MAX_CHUNK_TOKENS})."
        )
    if CHUNK_OVERLAP_TOKENS < 0:
        errors.append(f"CHUNK_OVERLAP_TOKENS={CHUNK_OVERLAP_TOKENS} must be >= 0.")
    if CHUNK_OVERLAP_TOKENS >= MAX_CHUNK_TOKENS:
        errors.append(
            f"CHUNK_OVERLAP_TOKENS ({CHUNK_OVERLAP_TOKENS}) must be < MAX_CHUNK_TOKENS ({MAX_CHUNK_TOKENS})."
        )

    # Layer 3
    if MAX_SUMMARY_LENGTH < 1:
        errors.append(f"MAX_SUMMARY_LENGTH={MAX_SUMMARY_LENGTH} must be >= 1.")
    if MIN_SUMMARY_LENGTH < 1:
        errors.append(f"MIN_SUMMARY_LENGTH={MIN_SUMMARY_LENGTH} must be >= 1.")

    # Layer 5
    if DEFAULT_TOP_K < 1:
        errors.append(f"DEFAULT_TOP_K={DEFAULT_TOP_K} must be >= 1.")
    if not (0.0 <= SIMILARITY_THRESHOLD <= 1.0):
        errors.append(
            f"SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD} must be between 0.0 and 1.0."
        )

    if errors:
        print("\n❌ Configuration errors detected:\n", file=sys.stderr)
        for e in errors:
            print(f"   • {e}", file=sys.stderr)
        print(
            "\n   Fix these in config.py or via environment variables.\n",
            file=sys.stderr,
        )
        sys.exit(1)


# Run validation on module import
_validate_config()
