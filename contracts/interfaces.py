"""
contracts/interfaces.py — Abstract Base Classes for all layers.

Architecture Principle: Plug-and-Play Modules
    "Modules can be replaced or upgraded without impacting the rest of the system."

Every layer has a concrete class (e.g., SpeechProcessor) that implements
its abstract interface (e.g., BaseSpeechProcessor). To swap a layer:
    1. Subclass the ABC
    2. Implement all abstract methods
    3. Pass the new class to the pipeline

The ABC enforces the output contract — you can't accidentally break downstream layers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from contracts.schemas import (
    Chunk,
    IntelligenceOutput,
    MeetingRecord,
    OutputResponse,
    QueryInput,
    RetrievedChunk,
    SpeechSegment,
)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Speech Processing
# ─────────────────────────────────────────────────────────────────────────────

class BaseSpeechProcessor(ABC):
    """
    Any ASR engine must implement this interface.

    Current implementation: SpeechProcessor (WhisperX)
    Future swaps:  Google Speech-to-Text, Azure STT, AssemblyAI, etc.
    """

    @abstractmethod
    def process(self, audio_path: str) -> List[SpeechSegment]:
        """
        Transform audio into speaker-aware transcript segments.

        Args:
            audio_path: Absolute path to an audio file.

        Returns:
            List[SpeechSegment] — the Layer 1 output contract.
            Must return empty list on failure (never raise).
        """
        ...

    def unload(self) -> None:
        """Release models from memory. Optional override."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Data Structuring
# ─────────────────────────────────────────────────────────────────────────────

class BaseStructurer(ABC):
    """
    Any chunking strategy must implement this interface.

    Current implementation: DataStructurer (token-bounded)
    Future swaps:  SemanticStructurer, SpeakerTurnStructurer, etc.
    """

    @abstractmethod
    def structure(
        self,
        segments: List[SpeechSegment],
        meeting_record: MeetingRecord,
    ) -> List[Chunk]:
        """
        Group segments into structured, metadata-enriched chunks.

        Args:
            segments:       Layer 1 output.
            meeting_record: Meeting metadata.

        Returns:
            List[Chunk] — the Layer 2 output contract.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Intelligence
# ─────────────────────────────────────────────────────────────────────────────

class BaseIntelligenceExtractor(ABC):
    """
    Any intelligence engine must implement this interface.

    Current implementation: IntelligenceExtractor (BART + spaCy)
    Future swaps:  GPT4Extractor, GeminiExtractor, LlamaExtractor, etc.
    """

    @abstractmethod
    def extract(self, chunks: List[Chunk], meeting_id: str) -> IntelligenceOutput:
        """
        Extract insights (summary, entities, intents) from chunks.

        Args:
            chunks:     Layer 2 output.
            meeting_id: Parent meeting identifier.

        Returns:
            IntelligenceOutput — the Layer 3 output contract.
        """
        ...

    def unload(self) -> None:
        """Release models from memory. Optional override."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Storage
# ─────────────────────────────────────────────────────────────────────────────

class BaseStore(ABC):
    """
    Any storage backend must implement this interface.

    Current implementation: MeetingStore (ChromaDB + SQLite)
    Future swaps:  PostgresStore, PineconeStore, WeaviateStore, etc.
    """

    @abstractmethod
    def save_meeting(self, record: MeetingRecord) -> None:
        ...

    @abstractmethod
    def save_chunks_with_embeddings(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        ...

    @abstractmethod
    def save_intelligence(self, output: IntelligenceOutput) -> None:
        ...

    @abstractmethod
    def get_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_meetings(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_intelligence(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def delete_meeting(self, meeting_id: str) -> bool:
        ...

    def close(self) -> None:
        """Release connections. Optional override."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — Query & Retrieval
# ─────────────────────────────────────────────────────────────────────────────

class BaseRetriever(ABC):
    """
    Any retrieval engine must implement this interface.

    Current implementation: SemanticRetriever (sentence-transformers + ChromaDB)
    Future swaps:  HybridRetriever (BM25 + dense), ColBERTRetriever, etc.
    """

    @abstractmethod
    def retrieve(self, query_input: QueryInput) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a user query.

        Args:
            query_input: QueryInput contract.

        Returns:
            List[RetrievedChunk] — sorted by descending relevance.
        """
        ...

    @abstractmethod
    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed chunk texts for storage."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 — Output Generation
# ─────────────────────────────────────────────────────────────────────────────

class BaseOutputGenerator(ABC):
    """
    Any answer generator must implement this interface.

    Current implementation: OutputGenerator (template-based)
    Future swaps:  LLMOutputGenerator (Ollama/OpenAI/Gemini), etc.
    """

    @abstractmethod
    def generate(
        self,
        query_input: QueryInput,
        chunks: List[RetrievedChunk],
    ) -> OutputResponse:
        """
        Generate a grounded answer with source attribution.

        Args:
            query_input: Original query.
            chunks:      Layer 5 output — retrieved chunks.

        Returns:
            OutputResponse — the Layer 6 output contract.
        """
        ...
