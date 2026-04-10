import json
import time
from typing import List
import os

import litellm

import config
from contracts.interfaces import BaseIntelligenceExtractor
from contracts.schemas import Chunk, IntelligenceOutput
from utils.logger import get_layer_logger

logger = get_layer_logger("intelligence")


class IntelligenceExtractor(BaseIntelligenceExtractor):
    """
    Stage 2 Intelligence: Structured LLM Extraction with Fallback

    - Uses Gemini when API key is available
    - Falls back to simple summarization if not
    - Prevents pipeline crashes
    """

    def __init__(self):
        logger.info("IntelligenceExtractor initialized (LLM + Fallback Mode).")

    def extract(self, chunks: List[Chunk], meeting_id: str) -> IntelligenceOutput:
        t0 = time.time()
        logger.info(f"Extracting structured intelligence for '{meeting_id}'...")

        full_text = "\n\n".join(
            f"[Speaker: {', '.join(c.speakers)}] {c.pure_text or c.text}"
            for c in chunks
        )

        if len(full_text) > 200000:
            logger.warning("Transcript too long, truncating...")
            full_text = full_text[:200000]

        if len(full_text.split()) < 20:
            return IntelligenceOutput(
                meeting_id=meeting_id,
                summary="Insufficient data.",
                entities=[],
                intents=[]
            )

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            logger.warning("No GEMINI_API_KEY found. Using fallback summarization.")

            fallback_summary = (
                full_text[:300] + "..." if len(full_text) > 300 else full_text
            )

            return IntelligenceOutput(
                meeting_id=meeting_id,
                summary=fallback_summary,
                entities=[],
                intents=[]
            )

        prompt = f"""
You are an expert meeting analyst.

Extract structured intelligence from this transcript.

Return STRICT JSON ONLY (no markdown, no explanation).

Format:
{{
  "summary": "2-3 sentence summary",
  "decisions": ["clear decisions"],
  "action_items": ["task (Owner: name, Deadline: if mentioned)"],
  "entities": ["important names, projects"]
}}

Rules:
- Do NOT hallucinate
- Only extract what is explicitly mentioned
- If missing, return empty []

TRANSCRIPT:
{full_text}
"""

        try:
            logger.info(f"Calling LLM ({config.LLM_MODEL})...")

            response = litellm.completion(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                api_key=api_key,
            )

            raw_output = response.choices[0].message.content.strip()

            if raw_output.startswith("```"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

            try:
                extracted_data = json.loads(raw_output)
            except json.JSONDecodeError:
                logger.warning("Retrying JSON parse after cleanup...")
                try:
                    raw_output = raw_output.replace("\n", " ").strip()
                    extracted_data = json.loads(raw_output)
                except json.JSONDecodeError:
                    logger.error(f"Final JSON parse failed:\n{raw_output}")
                    extracted_data = {
                        "summary": "Extraction failed due to formatting.",
                        "decisions": [],
                        "action_items": [],
                        "entities": []
                    }

            intents = extracted_data.get("decisions", []) + extracted_data.get("action_items", [])

            output = IntelligenceOutput(
                meeting_id=meeting_id,
                summary=extracted_data.get("summary", "Analysis complete."),
                entities=extracted_data.get("entities", []),
                intents=intents[:10],  # prevent DB bloat
            )

        except Exception as e:
            logger.error(f"LLM Extraction failed: {e}")

            fallback_summary = (
                full_text[:300] + "..." if len(full_text) > 300 else full_text
            )

            output = IntelligenceOutput(
                meeting_id=meeting_id,
                summary=fallback_summary,
                entities=[],
                intents=[]
            )

        elapsed = round(time.time() - t0, 2)
        logger.info(f"Extraction complete in {elapsed}s.")
        return output

    def unload(self):
        pass