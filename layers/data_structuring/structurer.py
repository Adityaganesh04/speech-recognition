import time
from typing import List, Optional, Set

import spacy
import config

from contracts.interfaces import BaseStructurer
from contracts.schemas import Chunk, MeetingRecord, SpeechSegment
from utils.logger import get_layer_logger

logger = get_layer_logger("data_structuring")


class SemanticStructurer(BaseStructurer):
    """
    Stage 2 Structurer:
    - Sentence-aware (spaCy)
    - Speaker-aware (hard boundaries)
    - No mid-sentence splits
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.max_tokens = config.MAX_CHUNK_TOKENS
        self.min_tokens = config.MIN_CHUNK_TOKENS

        logger.info("SemanticStructurer initialized (sentence + speaker aware)")

    def structure(
        self,
        segments: List[SpeechSegment],
        meeting_record: MeetingRecord,
    ) -> List[Chunk]:

        t0 = time.time()
        chunks: List[Chunk] = []

        chunk_text = ""
        chunk_speakers: Set[str] = set()
        chunk_start: Optional[float] = None
        chunk_end: Optional[float] = None

        chunk_index = 0
        prev_speaker = None

        for seg in segments:

            sentences = list(self.nlp(seg.text).sents)

            for sent in sentences:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                sent_words = len(sent_text.split())

                if prev_speaker is not None and seg.speaker != prev_speaker:
                    if chunk_text:
                        chunk = self._create_chunk(
                            chunk_text,
                            chunk_speakers,
                            chunk_start,
                            chunk_end,
                            meeting_record,
                            chunk_index,
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1

                    chunk_text = ""
                    chunk_speakers = set()
                    chunk_start = None
                    chunk_end = None

                current_words = len(chunk_text.split())

                if current_words + sent_words > self.max_tokens:
                    if chunk_text:
                        chunk = self._create_chunk(
                            chunk_text,
                            chunk_speakers,
                            chunk_start,
                            chunk_end,
                            meeting_record,
                            chunk_index,
                        )
                        if chunk:
                            chunks.append(chunk)
                            chunk_index += 1

                    chunk_text = ""
                    chunk_speakers = set()
                    chunk_start = None

                if not chunk_text:
                    chunk_start = seg.start

                chunk_text += " " + sent_text
                chunk_speakers.add(seg.speaker)
                chunk_end = seg.end

                prev_speaker = seg.speaker

        if chunk_text:
            chunk = self._create_chunk(
                chunk_text,
                chunk_speakers,
                chunk_start,
                chunk_end,
                meeting_record,
                chunk_index,
            )
            if chunk:
                chunks.append(chunk)

        logger.info(f"Semantic structuring complete: {len(chunks)} chunks")
        return chunks

    def _create_chunk(
        self,
        text: str,
        speakers: Set[str],
        start: float,
        end: float,
        meeting_record: MeetingRecord,
        index: int,
    ) -> Optional[Chunk]:

        text = text.strip()
        word_count = len(text.split())

        if word_count < self.min_tokens:
            return None

        return Chunk(
            chunk_id=f"{meeting_record.meeting_id}_c{index}",
            meeting_id=meeting_record.meeting_id,
            text=text,
            pure_text=text,
            start=round(start or 0.0, 3),
            end=round(end or 0.0, 3),
            speakers=sorted(list(speakers)),
            index=index
        )