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
        Includes automatic retry with backoff for rate-limited free-tier API keys.
        """
        # If the user toggles back to "template" in config, use the Stage 1 behavior
        if config.FEATURES.get("output_mode", "llm").lower() == "template":
            return self._generate_template_fallback(query, chunks)

        prompt = self._build_prompt(query, chunks)
        
        import os
        max_retries = 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating LLM response using model: '{config.LLM_MODEL}' (attempt {attempt + 1}/{max_retries})")
                response = litellm.completion(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    api_key=os.getenv("GEMINI_API_KEY"),
                    stream=False
                )
                
                answer = response.choices[0].message.content
                return answer

            except Exception as exc:
                error_str = str(exc).lower()
                
                # Check if it's a rate limit error
                if "429" in str(exc) or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    if attempt < max_retries - 1:
                        import time as _time
                        wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                        logger.warning(f"Rate limited by Gemini API. Waiting {wait_time}s before retry...")
                        _time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Gemini API rate limit exhausted after {max_retries} retries.")
                        return (
                            "⚠️ **Gemini API Rate Limit Reached**\n\n"
                            "Your free-tier Gemini API key has hit its daily quota (20 requests/day for gemini-2.5-flash). "
                            "The AI-powered analysis is temporarily unavailable.\n\n"
                            "**Options to resolve this:**\n"
                            "1. **Wait** — the quota resets automatically (check https://ai.dev/rate-limit)\n"
                            "2. **Upgrade** — enable billing on your Google AI Studio project for higher limits\n"
                            "3. **New API Key** — generate a fresh key at https://aistudio.google.com/apikey\n\n"
                            "Your question has been received and the transcript data is available. "
                            "Please try again once the rate limit resets."
                        )
                else:
                    logger.error(f"LLM Generation failed (non-rate-limit): {exc}")
                    return f"⚠️ **AI Generation Error:** {exc}"
        
        return "⚠️ Unexpected error in LLM generation."

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
        Builds a rich, detailed LLM prompt from retrieved chunks.
        Instructs Gemini to synthesize a thorough answer grounded in the transcript.
        """
        context_parts = []
        for c in chunks:
            speakers = ', '.join(c.speakers) if c.speakers else 'Unknown'
            context_parts.append(f"[Speaker: {speakers}]\n{c.text}")
        context = "\n\n---\n\n".join(context_parts)

        return (
            "You are an Expert Meeting Intelligence Analyst. Your role is to provide "
            "comprehensive, detailed, and insightful answers based EXCLUSIVELY on the "
            "meeting transcript excerpts provided below.\n\n"
            "INSTRUCTIONS:\n"
            "- Provide a thorough and detailed answer. Do NOT be vague or overly brief.\n"
            "- Include specific details, names, numbers, decisions, and action items mentioned by speakers.\n"
            "- Structure your response clearly using markdown (headings, bullet points, bold text) for readability.\n"
            "- Attribute information to specific speakers where possible (e.g., 'Speaker X mentioned that...').\n"
            "- Do NOT reference internal system identifiers like chunk IDs or similarity scores.\n"
            "- If the transcript excerpts do not contain enough information to fully answer the question, "
            "clearly state what IS available and what is missing.\n\n"
            f"USER QUESTION: {query}\n\n"
            f"=== MEETING TRANSCRIPT EXCERPTS ===\n\n{context}"
        )
