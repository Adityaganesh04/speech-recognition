"""
layers/data_structuring/structurer.py — Layer 2: Data Structuring

Responsibility:
    Convert a list of SpeechSegments into enriched, query-ready Chunk objects.

Pipeline:
    List[SpeechSegment] + MeetingRecord → List[Chunk]

Architecture principles applied:
    - Plug-and-play: inherits BaseStructurer ABC
    - Output contract: always returns List[Chunk]
    - Additive schema: new metadata fields added to Chunk without breaking callers
    - Observability: chunk count, word stats logged + metrics recorded
    - Chunk overlap: configurable overlap prevents context loss at boundaries
"""

import time
from typing import List, Optional, Set

import config
from contracts.interfaces import BaseStructurer
from contracts.schemas import Chunk, MeetingRecord, SpeechSegment
from utils.logger import get_layer_logger
from utils.metrics import metrics

logger = get_layer_logger("data_structuring")


class DataStructurer(BaseStructurer):
    """
    Converts raw speech segments into structured, metadata-enriched chunks.

    Chunking strategy (Stage 1):
        - Flush a chunk when adding the next segment would exceed max_tokens.
        - Discard chunks below min_tokens (noise/artifacts).
        - Chunks carry all speakers present in that window.
        - Adjacent chunks overlap by overlap_tokens words for context continuity.

    Upgrade path:
        - Replace _should_flush() for semantic boundary detection (Stage 2).
        - Add speaker role tagging without changing the Chunk schema.
    """

    def __init__(
        self,
        max_tokens: int = config.MAX_CHUNK_TOKENS,
        min_tokens: int = config.MIN_CHUNK_TOKENS,
        overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        logger.info(
            f"DataStructurer initialized "
            f"(max_tokens={max_tokens}, min_tokens={min_tokens}, "
            f"overlap={overlap_tokens})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def structure(
        self,
        segments: List[SpeechSegment],
        meeting_record: MeetingRecord,
    ) -> List[Chunk]:
        """
        Groups segments into token-bounded, metadata-enriched chunks
        with configurable overlap for context continuity.

        Args:
            segments:       Layer 1 output — list of SpeechSegment objects.
            meeting_record: Meeting metadata to attach to each chunk.

        Returns:
            List[Chunk] — the Layer 2 output contract.
        """
        t0 = time.time()
        logger.info(
            f"Structuring {len(segments)} segments "
            f"for meeting '{meeting_record.meeting_id}'..."
        )

        chunks: List[Chunk] = []
        buffer_texts: List[str] = []
        buffer_speakers: Set[str] = set()
        buffer_start: Optional[float] = None
        buffer_end: Optional[float] = None
        chunk_index = 0

        # Overlap carry-over from prior chunk
        overlap_prefix: str = ""
        overlap_speakers: Set[str] = set()

        for seg in segments:
            incoming_words = len(seg.text.split())

            # Flush buffer when adding this segment would exceed the token limit
            if buffer_texts and self._should_flush(buffer_texts, incoming_words):
                chunk, overlap_prefix, overlap_speakers = self._build_chunk_with_overlap(
                    buffer_texts,
                    buffer_speakers,
                    buffer_start,
                    buffer_end,
                    meeting_record,
                    chunk_index,
                    overlap_prefix,
                    overlap_speakers,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

                # Reset buffer
                buffer_texts = []
                buffer_speakers = set()
                buffer_start = None
                buffer_end = None

            # Accumulate into buffer
            buffer_texts.append(seg.text)
            buffer_speakers.add(seg.speaker)
            if buffer_start is None:
                buffer_start = seg.start
            buffer_end = seg.end

        # Flush final buffer
        if buffer_texts:
            chunk, _, _ = self._build_chunk_with_overlap(
                buffer_texts,
                buffer_speakers,
                buffer_start,
                buffer_end,
                meeting_record,
                chunk_index,
                overlap_prefix,
                overlap_speakers,
            )
            if chunk:
                chunks.append(chunk)

        elapsed = round(time.time() - t0, 3)
        total_words = sum(c.word_count for c in chunks)
        logger.info(
            f"Structuring complete: {len(chunks)} chunks, "
            f"{total_words} total words, {elapsed}s."
        )
        metrics.record("data_structuring", "structure", duration=elapsed, 
                       metadata={"chunks": len(chunks), "total_words": total_words})
        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _should_flush(self, buffer_texts: List[str], incoming_words: int) -> bool:
        """Return True if adding incoming_words would exceed the token limit."""
        current_words = len(" ".join(buffer_texts).split())
        return (current_words + incoming_words) > self.max_tokens

    def _build_chunk_with_overlap(
        self,
        texts: List[str],
        speakers: Set[str],
        start: Optional[float],
        end: Optional[float],
        meeting_record: MeetingRecord,
        index: int,
        overlap_prefix: str,
        overlap_speakers: Set[str],
    ) -> tuple:
        """
        Build a chunk, prepending overlap from the previous chunk.
        Returns (chunk_or_None, new_overlap_text, new_overlap_speakers).
        """
        raw_text = " ".join(texts).strip()

        # Prepend overlap from the prior chunk for context continuity
        if overlap_prefix and index > 0:
            full_text = overlap_prefix + " " + raw_text
            all_speakers = speakers | overlap_speakers
        else:
            full_text = raw_text
            all_speakers = speakers

        word_count = len(full_text.split())

        # Compute overlap to carry forward to the next chunk
        raw_words = raw_text.split()
        if self.overlap_tokens > 0 and len(raw_words) > self.overlap_tokens:
            new_overlap = " ".join(raw_words[-self.overlap_tokens:])
            new_overlap_speakers = set(speakers)
        else:
            new_overlap = ""
            new_overlap_speakers = set()

        if word_count < self.min_tokens:
            logger.debug(
                f"Chunk {index} discarded — too short "
                f"({word_count} words < {self.min_tokens})."
            )
            return None, new_overlap, new_overlap_speakers

        chunk = Chunk(
            chunk_id=f"{meeting_record.meeting_id}_c{index}",
            text=full_text,
            pure_text=raw_text,
            speakers=sorted(list(all_speakers)),
            meeting_id=meeting_record.meeting_id,
            language=meeting_record.language,
            start=round(start or 0.0, 3),
            end=round(end or 0.0, 3),
            word_count=word_count,
        )

        return chunk, new_overlap, new_overlap_speakers
