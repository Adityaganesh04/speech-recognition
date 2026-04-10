"""
layers/output/generator.py — Layer 6: Output Generation

Responsibility:
    Combine retrieved chunks into a user-facing answer with source attribution.

Pipeline:
    QueryInput + List[RetrievedChunk] → OutputResponse

Architecture principles applied:
    - Plug-and-play: _generate() is the single swap point for LLM integration
    - Output contract: always returns OutputResponse
    - Stage 1: context-grounded template answer (no external API needed)
    - Stage 2 upgrade: replace _generate() with an LLM call (Ollama / OpenAI / Gemini)
    - Observability: confidence, source count, and answer length are logged
"""

import time
from typing import List

import litellm

import config
from contracts.interfaces import BaseOutputGenerator
from contracts.schemas import OutputResponse, QueryInput, RetrievedChunk
from utils.logger import get_layer_logger

logger = get_layer_logger("output")


class OutputGenerator(BaseOutputGenerator):
    """
    Grounded answer generator.

    Stage 1 strategy:
        Format the top retrieved chunks as structured context with speaker labels
        and similarity scores. Source chunk IDs are always attributed.

    Upgrade path (Stage 2):
        Replace _generate() with:
            - ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
            - openai.chat.completions.create(...)
            - google.generativeai.generate_text(...)
        No other code changes required.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query_input: QueryInput,
        chunks: List[RetrievedChunk],
        stream_to_stdout: bool = True
    ) -> OutputResponse:
        """
        Build a grounded answer from retrieved chunks.

        Args:
            query_input: Original QueryInput contract.
            chunks:      Layer 5 output — ranked list of RetrievedChunk objects.

        Returns:
            OutputResponse — the Layer 6 output contract.
        """
        t0 = time.time()

        if not chunks:
            logger.info("No chunks retrieved — returning empty answer.")
            return OutputResponse(
                query=query_input.query,
                answer=(
                    "No relevant information was found for your query "
                    "in the stored meetings. Try rephrasing or ingesting more audio."
                ),
                sources=[],
                confidence=0.0,
            )

        answer = self._generate(query_input.query, chunks, stream_to_stdout)
        confidence = round(sum(c.score for c in chunks) / len(chunks), 4)
        sources = [c.chunk_id for c in chunks]

        elapsed = round(time.time() - t0, 3)
        logger.info(
            f"Answer generated in {elapsed}s — "
            f"confidence={confidence}, sources={len(sources)}, "
            f"answer_length={len(answer)} chars."
        )

        return OutputResponse(
            query=query_input.query,
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    def _generate(self, query: str, chunks: List[RetrievedChunk], stream_to_stdout: bool = True) -> str:
        """
        Stage 2: LLM-backed conversational RAG via litellm.
        """
        # If the user toggles back to "template" in config, use the Stage 1 behavior
        if config.FEATURES.get("output_mode", "llm").lower() == "template":
            return self._generate_template_fallback(query, chunks)

        prompt = self._build_prompt(query, chunks)
        
        try:
            import os
            import sys
            logger.info(f"Generating LLM response using model: '{config.LLM_MODEL}'")
            response = litellm.completion(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, # Keep output strictly grounded
                api_key=os.getenv("GEMINI_API_KEY"), # Explicitly pass key
                stream=True
            )
            
            if stream_to_stdout:
                print() # Start streaming on a new line
            pieces = []
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    if stream_to_stdout:
                        sys.stdout.write(content)
                        sys.stdout.flush()
                    pieces.append(content)
            
            return "".join(pieces)

        except Exception as exc:
            logger.error(f"LLM Generation failed: {exc}. Falling back to template.")
            return self._generate_template_fallback(query, chunks)

    def _generate_template_fallback(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Original Stage 1 Template Generator"""
        lines: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            speakers = ", ".join(chunk.speakers) if chunk.speakers else "Unknown"
            meeting = chunk.meeting_id
            score_pct = f"{chunk.score * 100:.1f}%"
            lines.append(
                f"[{idx}] Meeting: {meeting} | Speaker(s): {speakers} | Relevance: {score_pct}\n"
                f"    \"{chunk.text}\""
            )

        context_block = "\n\n".join(lines)
        source_ids = ", ".join(c.chunk_id for c in chunks)

        return (
            f'Query: "{query}"\n\n'
            f"Most relevant excerpts from meeting transcripts:\n\n"
            f"{context_block}\n\n"
            f"─────────────────────────────────────────────\n"
            f"Sources: {source_ids}"
        )

    def _build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Helper: builds an LLM-ready prompt for Stage 2 upgrade.
        Not used in Stage 1 but ready for plug-in.
        """
        context = "\n\n".join(
            f"[Speaker: {', '.join(c.speakers)}] {c.text}" for c in chunks
        )
        return (
            f"You are an expert meeting analyst. "
            f"Answer the following question using ONLY the provided meeting transcript excerpts.\n\n"
            f"Question: {query}\n\n"
            f"Transcript excerpts:\n{context}\n\n"
            f"Answer concisely and cite the source chunk IDs where relevant."
        )
