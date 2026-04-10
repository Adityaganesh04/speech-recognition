"""
layers/output/insights.py — Layer 6b: Cross-Meeting Insights

Responsibility:
    Synthesize high-level intelligence from multiple meetings simultaneously.
    (Stage 3: The Memory Layer component).

Architecture:
    Accepts a list of intelligence records (summaries, intents) and 
    generates a globally aware report using the generative LLM.
"""

import os
import time
from typing import Any, Dict, List

import litellm

import config
from utils.logger import get_layer_logger

logger = get_layer_logger("insights")


class InsightsGenerator:
    """
    Synthesizes long-term memory across all stored meetings.
    """

    def generate_global_insights(
        self, all_intelligence: List[Dict[str, Any]], custom_prompt: str = None
    ) -> str:
        """
        Builds a cross-meeting report.
        
        Args:
            all_intelligence: List of dicts representing Meeting Intelligence.
            custom_prompt: Optional user framing (e.g., "Focus only on UI features").
        """
        if not all_intelligence:
            return "No meeting intelligence available to generate insights. Please ingest meetings first."

        t0 = time.time()
        logger.info(f"Generating cross-meeting insights across {len(all_intelligence)} meetings...")

        # 1. Compile the Knowledge Block
        context_lines = []
        for intel in all_intelligence:
            meeting_id = intel.get("meeting_id", "Unknown")
            summary = intel.get("summary", "No summary.")
            intents = ", ".join(intel.get("intents", [])) or "None"
            
            context_lines.append(f"--- MEETING: {meeting_id} ---")
            context_lines.append(f"Summary: {summary}")
            context_lines.append(f"Recorded Intents/Decisions: {intents}")
            context_lines.append("")

        memory_block = "\n".join(context_lines)

        # 2. Build the Synthesis Prompt
        base_instruction = (
            "You are an Executive AI Analyst reviewing the global memory of all organizational meetings.\n"
            "Analyze the meeting summaries and intents provided below, and generate a comprehensive cross-meeting report.\n"
            "Structure your report dynamically based on the context and contents of the meetings, choosing the best sections and headers to present the insights clearly.\n"
            "Base your answers EXCLUSIVELY on the provided meeting records."
        )

        user_focus = (
            f"\n\nUSER DIRECTIVE: focus the synthesis around this request: '{custom_prompt}'"
            if custom_prompt
            else ""
        )

        final_prompt = f"{base_instruction}{user_focus}\n\n=== GLOBAL MEETING MEMORY ===\n\n{memory_block}"

        # 3. Call LLM
        try:
            response = litellm.completion(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3, # Slightly higher temperature (0.3 instead of 0.1) for synthesis mapping
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            report = response.choices[0].message.content
        except Exception as exc:
            logger.error(f"Global insights generation failed: {exc}")
            report = "Error: Failed to connect to the generative AI provider to synthesize insights."

        elapsed = round(time.time() - t0, 3)
        logger.info(f"Cross-meeting insights synthesized in {elapsed}s.")
        return report

    def generate_single_meeting_insights(
        self, meeting_id: str, chunks: List[Dict[str, Any]], custom_prompt: str = None
    ) -> str:
        """
        Builds a comprehensive, detailed report for a single meeting
        using its raw transcript chunks.
        """
        if not chunks:
            return f"No transcript data available for meeting '{meeting_id}'."

        t0 = time.time()
        logger.info(f"Generating detailed insights for meeting '{meeting_id}' ({len(chunks)} chunks)...")

        # 1. Compile the Transcript Block
        context_lines = []
        for c in chunks:
            speakers = ", ".join(c.get("speakers", [])) or "Unknown"
            # Use pure_text to avoid overlap stuttering, fallback to text for older chunks
            text = c.get("pure_text") or c.get("text", "")
            context_lines.append(f"[{speakers}]: {text}")

        transcript_block = "\n".join(context_lines)

        # Truncate if insanely long to avoid blowing up context window (Gemini 2.5 Flash has 1M context, so it should be fine, but we'll safeguard slightly)
        if len(transcript_block) > 300000:
            transcript_block = transcript_block[:300000] + "\n...[TRANSCRIPT TRUNCATED DUE TO LENGTH]..."

        # 2. Build the Synthesis Prompt
        base_instruction = (
            f"You are an Expert AI Meeting Analyst extracting exhaustive intelligence from a specific meeting transcript (ID: {meeting_id}).\n"
            "Your task is to provide an incredibly detailed and comprehensive summary of everything discussed, so that a reader would not need to listen to the audio to know exactly what was covered.\n"
            "Ensure you capture nuances, specific arguments, counter-arguments, actionable items, decisions, and specific details mentioned by different speakers.\n"
            "Structure your report dynamically using markdown, choosing the best sections and headers to present the insights clearly and comprehensively.\n"
            "Base your answers EXCLUSIVELY on the provided transcript."
        )

        user_focus = (
            f"\n\nUSER DIRECTIVE: Focus specifically on this request: '{custom_prompt}'"
            if custom_prompt
            else ""
        )

        final_prompt = f"{base_instruction}{user_focus}\n\n=== RAW TRANSCRIPT LOG ===\n\n{transcript_block}"

        # 3. Call LLM
        try:
            response = litellm.completion(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2, # Low temperature for factual extraction
                api_key=os.getenv("GEMINI_API_KEY"),
                stream=True
            )
            
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
        except Exception as exc:
            logger.error(f"Single meeting insights generation failed: {exc}")
            yield f"Error: Failed to connect to the generative AI provider to synthesize detailed meeting insights. {exc}"

        elapsed = round(time.time() - t0, 3)
        logger.info(f"Detailed meeting insights synthesized in {elapsed}s.")
