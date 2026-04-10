"""
contracts/schemas.py — Versioned Data Contracts
All inter-layer communication uses these Pydantic models.
Architecture principle: Strict interfaces, backward-compatible, schema-versioned.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 Output — Speech Processing
# ─────────────────────────────────────────────────────────────────────────────
class SpeechSegment(BaseModel):
    """Output contract for a single diarized transcript segment."""
    version: str = "v1"
    speaker: str                      # e.g. "SPEAKER_00"
    text: str                         # Transcribed text for this segment
    start: float                      # Start time in seconds
    end: float                        # End time in seconds


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 Output — Data Structuring
# ─────────────────────────────────────────────────────────────────────────────
class Chunk(BaseModel):
    chunk_id: str
    meeting_id: str
    text: str
    pure_text: str = ""
    speakers: List[str]
    
    language: str = "en"
    start: float
    end: float
    word_count: int = 0

# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 Output — Intelligence
# ─────────────────────────────────────────────────────────────────────────────
class IntelligenceOutput(BaseModel):
    """Output contract for meeting-level intelligence extraction."""
    version: str = "v1"
    meeting_id: str
    summary: str                      # Global meeting summary
    entities: List[str] = []          # Named entities (people, orgs, dates, etc.)
    intents: List[str] = []           # Detected intents (Stage 2 expansion point)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Storage Unit (internal, used by storage layer)
# ─────────────────────────────────────────────────────────────────────────────
class StoredUnit(BaseModel):
    """Internal contract for a vector-stored chunk."""
    version: str = "v1"
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 I/O — Query & Retrieval
# ─────────────────────────────────────────────────────────────────────────────
class QueryInput(BaseModel):
    """Input contract for a user search query."""
    version: str = "v1"
    query: str                                  # Natural language query
    meeting_id: Optional[str] = None            # Scope to a specific meeting (None = all)
    top_k: int = 5                              # Number of results to retrieve


class RetrievedChunk(BaseModel):
    """Output contract for a single retrieved search result."""
    version: str = "v1"
    chunk_id: str
    text: str
    score: float                                # Cosine similarity score (0.0–1.0)
    speakers: List[str]
    meeting_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 Output — Output Generation
# ─────────────────────────────────────────────────────────────────────────────
class OutputResponse(BaseModel):
    """Output contract for a generated answer with source attribution."""
    version: str = "v1"
    query: str
    answer: str
    sources: List[str]                          # chunk_ids used to generate this answer
    confidence: float = 0.0                     # Average retrieval score


# ─────────────────────────────────────────────────────────────────────────────
# Meeting Record — Top-level entity stored in SQLite
# ─────────────────────────────────────────────────────────────────────────────
class MeetingRecord(BaseModel):
    """Full meeting metadata record."""
    version: str = "v1"
    meeting_id: str
    audio_path: str
    participants: List[str] = []
    language: str = "en"
    duration: float = 0.0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chunk_count: int = 0
    summary: str = ""
