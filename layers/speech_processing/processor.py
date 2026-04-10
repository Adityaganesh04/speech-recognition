"""
layers/speech_processing/processor.py — Layer 1: Speech Processing

Responsibility:
    Transform raw audio into a list of SpeechSegment objects.

Pipeline:
    Audio file → ASR (WhisperX) → Word Alignment → Speaker Diarization → SpeechSegments

Architecture principles applied:
    - Plug-and-play: swap ASR engine by subclassing or replacing SpeechProcessor
    - Output contract: always returns List[SpeechSegment]
    - Observability: every major step is logged with timing context
    - Fallback: diarization is optional (disabled when HF_TOKEN is not set)
    - Graceful degradation: each stage has independent error handling

Fixes applied (audit):
    - [#1] Try/except on every heavy operation (ASR, alignment, diarization)
    - [#2] Diarization receives audio_path (str), not numpy array
    - [#6] GPU memory cleanup after processing via gc + torch.cuda.empty_cache
    - [#12] Alignment model cache invalidated when language changes
"""

import gc
import time
from typing import List, Optional

import config
from contracts.interfaces import BaseSpeechProcessor
from contracts.schemas import SpeechSegment
from utils.logger import get_layer_logger
from utils.metrics import metrics

logger = get_layer_logger("speech_processing")


class SpeechProcessor(BaseSpeechProcessor):
    """
    Plug-and-play speech processor backed by WhisperX.

    Swap this class's internals to use any ASR engine without touching
    downstream layers — the output contract (List[SpeechSegment]) stays fixed.
    """

    def __init__(self):
        logger.info(
            f"Loading Whisper model '{config.WHISPER_MODEL}' "
            f"on device='{config.DEVICE}', compute_type='{config.COMPUTE_TYPE}'"
        )
        import whisperx
        self._whisperx = whisperx
        self.model = whisperx.load_model(
            config.WHISPER_MODEL,
            config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
        )
        self._align_model = None
        self._align_metadata = None
        self._align_language = None        # Track cached language for invalidation
        self._diarize_model = None
        logger.info("SpeechProcessor ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, audio_path: str) -> List[SpeechSegment]:
        """
        Full pipeline: ASR → Alignment → Diarization → SpeechSegments.

        Args:
            audio_path: Absolute or relative path to an audio file.

        Returns:
            List[SpeechSegment] — the Layer 1 output contract.
            Returns empty list on complete failure (never raises to caller).
        """
        t0 = time.time()
        logger.info(f"Starting speech processing: {audio_path}")

        try:
            audio = self._whisperx.load_audio(audio_path)
        except Exception as exc:
            logger.error(f"Failed to load audio file: {exc}")
            return []

        # Step 1 — ASR
        result = self._run_asr(audio)
        if result is None:
            return []
        language: str = result.get("language", "en")
        logger.info(f"ASR complete. Detected language: '{language}'")

        # Step 2 — Word-level alignment
        result = self._run_alignment(audio, result, language)
        if result is None:
            return []

        # Step 3 — Speaker diarization (optional)
        if config.DIARIZATION_ENABLED:
            result = self._run_diarization(audio_path, result)
            # Diarization failure is non-fatal — segments still usable without speaker labels
        else:
            logger.info(
                "Diarization DISABLED (set HF_TOKEN env var to enable). "
                "All segments will be labeled 'SPEAKER_00'."
            )

        # Step 4 — Build output contracts
        segments = self._build_segments(result["segments"])
        elapsed = round(time.time() - t0, 2)
        logger.info(f"Speech processing complete: {len(segments)} segments in {elapsed}s.")

        # Step 5 — Memory cleanup (prevents OOM across multiple ingestions)
        self._cleanup_memory()

        return segments

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_asr(self, audio) -> Optional[dict]:
        try:
            logger.info("Running ASR transcription...")
            return self.model.transcribe(audio, batch_size=16)
        except Exception as exc:
            logger.error(f"ASR transcription failed: {exc}")
            return None

    def _run_alignment(self, audio, result: dict, language: str) -> Optional[dict]:
        try:
            # Invalidate alignment cache if language changed (fix #12)
            if self._align_model is not None and self._align_language != language:
                logger.info(
                    f"Language changed ({self._align_language} → {language}). "
                    "Reloading alignment model..."
                )
                self._align_model = None
                self._align_metadata = None

            if self._align_model is None:
                logger.info(f"Loading alignment model for language '{language}'...")
                self._align_model, self._align_metadata = (
                    self._whisperx.load_align_model(
                        language_code=language,
                        device=config.DEVICE,
                    )
                )
                self._align_language = language

            logger.info("Aligning word-level timestamps...")
            return self._whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                config.DEVICE,
                return_char_alignments=False,
            )
        except Exception as exc:
            logger.error(f"Alignment failed: {exc}. Proceeding with unaligned segments.")
            return result  # Fall back to unaligned segments

    def _run_diarization(self, audio_path: str, result: dict) -> dict:
        """
        Run speaker diarization.

        Note: WhisperX DiarizationPipeline expects the audio FILE PATH (str),
        not a loaded numpy array. This was fixed from the original (audit #2).
        """
        try:
            if self._diarize_model is None:
                from whisperx.diarize import DiarizationPipeline
                logger.info("Loading speaker diarization model...")
                self._diarize_model = DiarizationPipeline(
                    token=config.HF_TOKEN,
                    device=config.DEVICE,
                )
            logger.info("Running speaker diarization...")
            diarize_segments = self._diarize_model(audio_path)
            return self._whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as exc:
            logger.warning(
                f"Diarization failed (likely HuggingFace network issue): {exc}. "
                "Continuing without speaker labels (all speakers → SPEAKER_00)."
            )
            return result  # Non-fatal: proceed without speaker separation

    def _build_segments(self, raw_segments: list) -> List[SpeechSegment]:
        """Convert raw WhisperX segments into validated SpeechSegment contracts."""
        output: List[SpeechSegment] = []
        for seg in raw_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            output.append(
                SpeechSegment(
                    speaker=seg.get("speaker", "SPEAKER_00"),
                    text=text,
                    start=round(float(seg.get("start", 0.0)), 3),
                    end=round(float(seg.get("end", 0.0)), 3),
                )
            )
        return output

    def _cleanup_memory(self):
        """Release GPU/CPU memory after processing to prevent OOM across runs."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory cache cleared.")
        except ImportError:
            pass

    def unload(self):
        """
        Explicitly release all models from memory.
        Call this when the processor is no longer needed (e.g., after batch ingestion).
        """
        logger.info("Unloading speech processing models...")
        self.model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None
        self._cleanup_memory()
        logger.info("Speech processing models unloaded.")
"""
    Layer 1 — Speech Processing

    Responsibilities:
    - ASR transcription via WhisperX
    - Word-level timestamp alignment
    - Optional speaker diarization (requires HF_TOKEN)

    Output Contract: List[SpeechSegment]
"""
