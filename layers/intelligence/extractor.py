"""
layers/intelligence/extractor.py — Layer 3: Intelligence

Responsibility:
    Extract meaning from structured chunks — summarization and named entities.

Pipeline:
    List[Chunk] → IntelligenceOutput

Architecture principles applied:
    - Model-agnostic: models are loaded lazily and can be swapped via config
    - Plug-and-play: _summarize() and _extract_entities() are independently replaceable
    - Output contract: always returns IntelligenceOutput
    - Observability: model load events and extraction stats are logged
    - Graceful degradation: regex fallback if spaCy is unavailable

Fixes applied (audit):
    - [#3] spaCy text length limit protection — chunk text for NER
    - [#7] Added unload() method for explicit memory release
"""

import gc
import re
import time
from typing import List, Optional

import config
from contracts.interfaces import BaseIntelligenceExtractor
from contracts.schemas import Chunk, IntelligenceOutput
from utils.logger import get_layer_logger
from utils.metrics import metrics

logger = get_layer_logger("intelligence")

# spaCy default max_length is 1M chars; we process in chunks below this
_SPACY_MAX_CHARS = 500_000


class IntelligenceExtractor(BaseIntelligenceExtractor):
    """
    Model-agnostic intelligence layer.

    Stage 1 capabilities:
        - Global summarization (BART)
        - Named entity extraction (spaCy or regex fallback)

    Upgrade path (Stage 2):
        - Add intent detection, task extraction, decision tracking
        - Swap BART for any HuggingFace or API-backed model via _summarize()
    """

    def __init__(self):
        # Lazy-load heavy models — don't block startup
        self._summarizer = None
        self._nlp = None
        logger.info(
            "IntelligenceExtractor initialized (models will load on first use)."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, chunks: List[Chunk], meeting_id: str) -> IntelligenceOutput:
        """
        Run summarization and entity extraction across all chunks.

        Args:
            chunks:     Layer 2 output — list of Chunk objects.
            meeting_id: Parent meeting identifier.

        Returns:
            IntelligenceOutput — the Layer 3 output contract.
        """
        t0 = time.time()
        logger.info(
            f"Running intelligence extraction for meeting '{meeting_id}' "
            f"({len(chunks)} chunks)..."
        )

        full_text = " ".join(c.text for c in chunks)

        # Feature flag: skip summarization if disabled
        if config.FEATURES.get("enable_summarization", True):
            summary = self._summarize(full_text)
        else:
            logger.info("Summarization disabled via feature flag.")
            words = full_text.split()
            summary = " ".join(words[:80]) + ("..." if len(words) > 80 else "")

        # Feature flag: skip entity extraction if disabled
        if config.FEATURES.get("enable_entity_extraction", True):
            entities = self._extract_entities(full_text)
        else:
            logger.info("Entity extraction disabled via feature flag.")
            entities = []

        # Feature flag: extract global meeting intents using the LLM
        intents = []
        if config.FEATURES.get("enable_intent_extraction", True):
            intents = self._extract_intents_from_summary(summary)
        else:
            logger.info("Intent extraction disabled via feature flag.")

        output = IntelligenceOutput(
            meeting_id=meeting_id,
            summary=summary,
            entities=entities,
            intents=intents,
        )

        elapsed = round(time.time() - t0, 2)
        logger.info(
            f"Intelligence extraction complete in {elapsed}s — "
            f"summary={len(summary)} chars, entities={len(entities)}, intents={len(intents)}."
        )
        return output

    # ── Summarization ─────────────────────────────────────────────────────────

    def _get_summarizer(self):
        if self._summarizer is None:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading summarization model: '{config.SUMMARIZATION_MODEL}'...")
            self._summarizer = hf_pipeline(
                "summarization",
                model=config.SUMMARIZATION_MODEL,
                device=-1,  # CPU (-1); change to 0 for first GPU
            )
            logger.info("Summarization model loaded.")
        return self._summarizer

    def _summarize(self, text: str) -> str:
        """Summarize text, truncating to avoid model input limits (~1024 tokens)."""
        if not text or not text.strip():
            logger.warning("Empty text provided for summarization.")
            return "No content to summarize."

        try:
            summarizer = self._get_summarizer()
            words = text.split()
            # BART input ceiling ~1024 tokens; use 900 words as safe limit
            if len(words) > 900:
                logger.info("Transcript truncated to 900 words for summarization.")
                text = " ".join(words[:900])

            # Guard against text too short for summarization
            if len(words) < 10:
                logger.info("Text too short for summarization model. Returning as-is.")
                return text

            result = summarizer(
                text,
                max_length=config.MAX_SUMMARY_LENGTH,
                min_length=min(config.MIN_SUMMARY_LENGTH, len(words)),
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"]

        except Exception as exc:
            logger.error(f"Summarization failed: {exc}. Using truncated transcript.")
            words = text.split()
            return " ".join(words[:80]) + ("..." if len(words) > 80 else "")

    # ── Entity Extraction ─────────────────────────────────────────────────────

    def _get_nlp(self):
        if self._nlp is None and config.USE_SPACY:
            try:
                import spacy

                logger.info(f"Loading spaCy model: '{config.SPACY_MODEL}'...")
                self._nlp = spacy.load(config.SPACY_MODEL)
                # Raise max_length to handle large transcripts
                self._nlp.max_length = 2_000_000
                logger.info("spaCy model loaded.")
            except Exception as exc:
                logger.warning(
                    f"spaCy load failed ({exc}). "
                    "Falling back to regex entity extraction."
                )
                self._nlp = None
        return self._nlp

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities using spaCy or capitalized noun phrase regex.

        Fix #3: For very long texts, process in chunks to avoid spaCy max_length crash.
        """
        if not text or not text.strip():
            return []

        nlp = self._get_nlp()
        entities: set = set()

        if nlp is not None:
            try:
                target_labels = {"PERSON", "ORG", "DATE", "TIME", "GPE", "PRODUCT", "EVENT"}

                # Process in safe-sized chunks to prevent spaCy ValueError
                text_chunks = self._split_text_for_nlp(text)
                for chunk_text in text_chunks:
                    doc = nlp(chunk_text)
                    for ent in doc.ents:
                        if ent.label_ in target_labels and len(ent.text.strip()) > 1:
                            entities.add(ent.text.strip())

            except Exception as exc:
                logger.warning(f"spaCy entity extraction failed: {exc}. Using regex fallback.")
                entities = self._regex_entities(text)
        else:
            entities = self._regex_entities(text)

        return sorted(list(entities))

    def _split_text_for_nlp(self, text: str) -> List[str]:
        """Split text into chunks safe for spaCy processing."""
        if len(text) <= _SPACY_MAX_CHARS:
            return [text]

        logger.info(
            f"Text length ({len(text)} chars) exceeds safe limit. "
            f"Splitting into {len(text) // _SPACY_MAX_CHARS + 1} chunks for NER."
        )
        chunks = []
        words = text.split()
        current_chunk: List[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space
            if current_len + word_len > _SPACY_MAX_CHARS and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(word)
            current_len += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _regex_entities(self, text: str) -> set:
        """Regex fallback for entity extraction."""
        matches = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text)
        stopwords = {"The", "This", "That", "We", "Our", "So", "But", "If", "I", "A"}
        result = {m for m in matches if m not in stopwords}
        return set(list(result)[:30])  # Cap at 30 results

    # ── Intent Extraction ─────────────────────────────────────────────────────

    def _extract_intents_from_summary(self, summary: str) -> List[str]:
        """
        Stage 2: Uses the global generative LLM to extract primary intents, 
        tasks, or decisions from the meeting's compiled summary.
        """
        if not summary or len(summary.strip()) < 20:
            return []

        try:
            import litellm
            import os
            
            prompt = (
                "You are an expert meeting analyst. Based on the following meeting summary, "
                "identify the 1 to 3 primary intents, action items, or core decisions discussed.\n"
                "Return them as a simple, comma-separated list. No bullet points or extra text.\n\n"
                f"Summary: {summary}"
            )
            logger.info(f"Extracting meeting intents via {config.LLM_MODEL}...")
            response = litellm.completion(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # Keep output strictly formatted
                api_key=os.getenv("GEMINI_API_KEY"), # Explicitly pass key
            )
            raw_intents = response.choices[0].message.content
            
            # Clean up comma separated list
            intents = [i.strip() for i in str(raw_intents).split(",") if i.strip()]
            return intents[:5]  # Cap at 5 intents

        except Exception as exc:
            logger.error(f"Intent extraction via LLM failed: {exc}. Verify LLM API key.")
            return []

    # ── Resource Management ───────────────────────────────────────────────────

    def unload(self):
        """Explicitly release models from memory."""
        logger.info("Unloading intelligence models...")
        self._summarizer = None
        self._nlp = None
        gc.collect()
        logger.info("Intelligence models unloaded.")
