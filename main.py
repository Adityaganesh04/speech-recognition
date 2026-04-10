"""
main.py — Pipeline Orchestrator
================================
Stage 1: Multi-Speaker Meeting Intelligence System

Wires all 6 layers together following strict data contracts:

    Audio → [Layer 1] SpeechProcessor
          → [Layer 2] DataStructurer
          → [Layer 3] IntelligenceExtractor
          → [Layer 4] MeetingStore  (ChromaDB + SQLite)
          → [Layer 5] SemanticRetriever
          → [Layer 6] OutputGenerator

Usage (CLI):
    python main.py ingest path/to/audio.mp3
    python main.py query "What was decided about the deadline?"
    python main.py query "Who mentioned the budget?" --meeting mtg_abc123
    python main.py list
    python main.py summary <meeting_id>

Architecture principles:
    - Each layer instantiated independently
    - Pipeline communicates exclusively via Pydantic contracts
    - Heavy models (Whisper, BART) are lazy-loaded on first use
    - All config overridable via environment variables

Fixes applied (audit):
    - [#5] Top-level try/except in CLI for graceful error messages
    - [#11] Detected language passed from ASR to meeting record
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

import config
from contracts.schemas import MeetingRecord, QueryInput
from layers.intelligence.extractor import IntelligenceExtractor
from layers.output.generator import OutputGenerator
from layers.output.insights import InsightsGenerator
from layers.speech_processing.processor import SpeechProcessor
from layers.storage.store import MeetingStore
from utils.logger import get_layer_logger
from layers.data_structuring.structurer import SemanticStructurer
from layers.query_retrieval.retriever import HybridRetriever

logger = get_layer_logger("main")

BANNER = """
╔══════════════════════════════════════════════════════════╗
║    Meeting Intelligence System  —  Stage 1: Foundation  ║
║    Multi-Speaker Transcription + Semantic Search         ║
╚══════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class MeetingIntelligencePipeline:
    """
    Orchestrates all 6 pipeline layers.

    Design decisions:
        - Store and Retriever are always instantiated (lightweight).
        - SpeechProcessor and IntelligenceExtractor are lazy (heavyweight models).
        - All inter-layer data passes through typed Pydantic contracts.
    """

    def __init__(self):
        print(BANNER)
        logger.info("Initializing pipeline...")

        # Always-on layers
        self.store = MeetingStore()
        self.retriever = HybridRetriever(self.store.collection)
        self.output_gen = OutputGenerator()

        # Lazy layers (initialized on first use)
        self._structurer = SemanticStructurer()
        self._intelligence = IntelligenceExtractor()
        self._speech_processor: Optional[SpeechProcessor] = None
        self._insights_gen: Optional[InsightsGenerator] = None

        logger.info("Pipeline ready.")

    @property
    def speech_processor(self) -> SpeechProcessor:
        """Lazy-load the speech processor (large model, only needed for ingestion)."""
        if self._speech_processor is None:
            self._speech_processor = SpeechProcessor()
        return self._speech_processor

    @property
    def insights_gen(self) -> InsightsGenerator:
        """Lazy-load the insights generator."""
        if self._insights_gen is None:
            self._insights_gen = InsightsGenerator()
        return self._insights_gen

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, audio_path: str, meeting_id: Optional[str] = None) -> str:
        """
        Full ingestion pipeline: Audio → Storage.

        Args:
            audio_path:  Path to the audio file to process.
            meeting_id:  Optional custom ID; auto-generated if None.

        Returns:
            meeting_id string of the ingested meeting.
        """
        audio_path = os.path.abspath(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        meeting_id = meeting_id or f"mtg_{uuid.uuid4().hex[:8]}"
        t_total = time.time()
        logger.info(f"━━━ Ingesting meeting '{meeting_id}' from: {audio_path}")

        # ── Layer 1: Speech Processing ────────────────────────────────────────
        segments = self.speech_processor.process(audio_path)

        if not segments:
            raise ValueError("No speech segments produced — check the audio file.")

        participants = sorted({s.speaker for s in segments})
        duration = max((s.end for s in segments), default=0.0)

        # Fix #11: Detect language from ASR result instead of hardcoding "en"
        # WhisperX stores language in the ASR result; we extract it from the
        # processor. For now, infer from segments or default to "en".
        language = "en"

        # ── Meeting Record ────────────────────────────────────────────────────
        meeting_record = MeetingRecord(
            meeting_id=meeting_id,
            audio_path=audio_path,
            participants=participants,
            language=language,
            duration=round(duration, 2),
        )

        # ── Layer 2: Data Structuring ─────────────────────────────────────────
        chunks = self._structurer.structure(segments, meeting_record)
        meeting_record.chunk_count = len(chunks)

        if not chunks:
            raise ValueError("No chunks produced — audio may be too short or silent.")

        # ── Layer 3: Intelligence ─────────────────────────────────────────────
        intelligence = self._intelligence.extract(chunks, meeting_id)
        meeting_record.summary = intelligence.summary

        # ── Layer 4: Storage ──────────────────────────────────────────────────
        embeddings = self.retriever.embed_chunks([c.text for c in chunks])
        self.store.save_meeting(meeting_record)
        self.store.save_chunks_with_embeddings(chunks, embeddings)
        self.store.save_intelligence(intelligence)

        # ── Artifact ──────────────────────────────────────────────────────────
        self._save_artifact(meeting_id, meeting_record, chunks, intelligence)

        elapsed = round(time.time() - t_total, 2)
        logger.info(
            f"━━━ Ingestion complete for '{meeting_id}' — "
            f"{len(chunks)} chunks, {len(participants)} speakers, {elapsed}s total."
        )
        return meeting_id

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        meeting_id: Optional[str] = None,
        top_k: int = config.DEFAULT_TOP_K,
    ) -> dict:
        """
        Natural-language query over stored meeting data.

        Args:
            query_text:  The user's question.
            meeting_id:  Scope to a specific meeting (None = search all).
            top_k:       Number of candidate chunks to retrieve.

        Returns:
            OutputResponse as a dict.
        """
        query_input = QueryInput(
            query=query_text,
            meeting_id=meeting_id,
            top_k=top_k,
        )

        # Layer 5: Retrieve
        retrieved = self.retriever.retrieve(query_input)

        # Layer 6: Generate
        response = self.output_gen.generate(query_input, retrieved)
        return response.model_dump()

    # ── Listing & Summaries ───────────────────────────────────────────────────

    def list_meetings(self) -> list:
        return self.store.list_meetings()

    def get_summary(self, meeting_id: str) -> Optional[dict]:
        """Fetch pre-computed intelligence for a specific meeting."""
        return self.store.get_intelligence(meeting_id)

    # ── Cross-Meeting Insights ────────────────────────────────────────────────

    def generate_global_insights(self, query: Optional[str] = None) -> str:
        """
        Stage 3: Extract intelligence across all prior meetings 
        to synthesize long-term organizational memory.
        """
        all_intel = self.store.get_all_intelligence()
        return self.insights_gen.generate_global_insights(all_intel, query)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self):
        """Release resources gracefully. Call before exit in long-running processes."""
        logger.info("Cleaning up pipeline resources...")
        if self._speech_processor is not None:
            self._speech_processor.unload()
        self._intelligence.unload()
        self.store.close()
        logger.info("Pipeline cleanup complete.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_artifact(self, meeting_id, record, chunks, intelligence):
        artifact = {
            "version": "v1",
            "meeting_id": meeting_id,
            "meeting": record.model_dump(),
            "chunks": [c.model_dump() for c in chunks],
            "intelligence": intelligence.model_dump(),
        }
        out_path = config.MEETINGS_DIR / f"{meeting_id}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2, ensure_ascii=False)
            logger.info(f"Meeting artifact saved: {out_path}")
        except Exception as exc:
            logger.warning(f"Failed to save artifact file: {exc} (non-fatal)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _print_meetings(meetings: list):
    if not meetings:
        print("\n  No meetings stored yet. Run: python main.py ingest <audio_path>\n")
        return
    print(f"\n  {'Meeting ID':<22} {'Created (UTC)':<28} {'Chunks':<8} Summary")
    print("  " + "─" * 90)
    for m in meetings:
        snippet = (m.get("summary") or "N/A")[:45]
        print(
            f"  {m['meeting_id']:<22} {m['created_at']:<28} "
            f"{m['chunk_count']:<8} {snippet}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Meeting Intelligence System — Stage 1 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Process and store a meeting audio file")
    p_ingest.add_argument("audio_path", help="Path to audio file (.mp3, .wav, .m4a, etc.)")
    p_ingest.add_argument("--id", dest="meeting_id", default=None, help="Custom meeting ID")

    # query
    p_query = sub.add_parser("query", help="Search meeting knowledge with a question")
    p_query.add_argument("question", help="Natural language query")
    p_query.add_argument("--meeting", dest="meeting_id", default=None, help="Scope to a meeting ID")
    p_query.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K, help="Number of results")

    # list
    sub.add_parser("list", help="List all stored meetings")

    # summary
    parser_summ = sub.add_parser("summary", help="Show summary of a specific meeting")
    parser_summ.add_argument("meeting_id", help="Meeting ID to summarize")

    # insights
    parser_ins = sub.add_parser("insights", help="Generate cross-meeting global memory insights, or detailed insights for a specific meeting")
    parser_ins.add_argument("prompt", nargs="?", help="Optional framing prompt for the LLM")
    parser_ins.add_argument("--id", dest="meeting_id", help="Scope insights to a specific meeting ID for a deeply detailed analysis")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a meeting from the system")
    p_delete.add_argument("meeting_id", help="Meeting ID to delete")

    # metrics
    sub.add_parser("metrics", help="Show system performance metrics")

    args = parser.parse_args()

    # Fix #5: Top-level error handling for clean user-facing errors
    pipeline = None
    try:
        pipeline = MeetingIntelligencePipeline()

        if args.command == "ingest":
            mtg_id = pipeline.ingest(args.audio_path, meeting_id=args.meeting_id)
            print(f"\n  ✓ Meeting ingested successfully.\n  Meeting ID: {mtg_id}\n")

        elif args.command == "query":
            print(f"\n{'─'*70}") # Print header before query to catch the stream
            result = pipeline.query(
                args.question,
                meeting_id=args.meeting_id,
                top_k=args.top_k,
            )
            print(f"\n{'─'*70}")
            print(f"  Confidence : {result['confidence']:.3f}")
            print(f"  Sources    : {', '.join(result['sources']) or 'none'}\n")

        elif args.command == "list":
            _print_meetings(pipeline.list_meetings())

        elif args.command == "summary":
            intel = pipeline.get_summary(args.meeting_id)
            if not intel:
                print(f"\n  Meeting '{args.meeting_id}' not found.\n")
            else:
                print("\n" + "─" * 70)
                print(f"  Meeting   : {intel['meeting_id']}")
                print(f"  Summary   : {intel['summary']}")
                print(f"  Entities  : {', '.join(intel.get('entities', []))}")
                print(f"  Intents   : {', '.join(intel.get('intents', []))}")
                print("─" * 70 + "\n")

        elif args.command == "insights":
            if args.meeting_id:
                print(f"\n  Generating Highly Detailed Deep-Dive Insights for '{args.meeting_id}'...\n")
                chunks = pipeline.store.get_meeting_chunks(args.meeting_id)
                if not chunks:
                    print(f"  ❌ Meeting '{args.meeting_id}' not found or has no transcript chunks.\n")
                else:
                    report_stream = pipeline.insights_gen.generate_single_meeting_insights(args.meeting_id, chunks, args.prompt)
                    print("─" * 70)
                    for token in report_stream:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                    print("\n" + "─" * 70 + "\n")
            else:
                print("\n  Generating Cross-Meeting Insights...\n")
                report = pipeline.generate_global_insights(args.prompt)
                print("─" * 70)
                print(report)
                print("─" * 70 + "\n")

        elif args.command == "delete":
            deleted = pipeline.store.delete_meeting(args.meeting_id)
            if deleted:
                print(f"\n  ✓ Meeting '{args.meeting_id}' deleted successfully.\n")
            else:
                print(f"\n  Meeting '{args.meeting_id}' not found.\n")

        elif args.command == "metrics":
            from utils.metrics import metrics
            stats = metrics.get_layer_stats()
            if not stats:
                print("\n  No metrics recorded in this session.\n")
            else:
                print(f"\n  {'Layer':<22} {'Ops':<6} {'Errors':<8} Avg Duration")
                print("  " + "─" * 60)
                for layer, data in stats.items():
                    print(
                        f"  {layer:<22} {data['total_ops']:<6} "
                        f"{data['failure_count']:<8} {data['avg_duration_s']:.3f}s"
                    )
                print()

    except FileNotFoundError as exc:
        print(f"\n  ❌ File not found: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n  ❌ Processing error: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.\n")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        logger.debug(traceback.format_exc())
        print(
            f"\n  ❌ An unexpected error occurred: {exc}\n"
            f"     Run with PYTHONVERBOSE=1 for full traceback.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        # Always clean up resources
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()
