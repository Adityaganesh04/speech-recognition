"""
utils/metrics.py — Lightweight Metrics Collector

Architecture Principle: Observability-Driven Design
    "Each layer is monitored for: Latency, Accuracy, Failure rates"

Provides structured performance metrics for every layer operation.
Queryable at runtime and exportable for dashboards (Stage 5).

Usage in layers:
    from utils.metrics import metrics
    metrics.record("speech_processing", "asr_transcribe", duration=12.5, success=True)
    metrics.record("intelligence", "summarize", duration=3.2, success=False, error="OOM")

Usage from CLI:
    python main.py metrics
"""

import time
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from utils.logger import get_layer_logger

logger = get_layer_logger("metrics")


class _OperationRecord:
    """Single recorded operation."""
    __slots__ = ("layer", "operation", "duration", "success", "error", "timestamp", "metadata")

    def __init__(
        self,
        layer: str,
        operation: str,
        duration: float,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.layer = layer
        self.operation = operation
        self.duration = duration
        self.success = success
        self.error = error
        self.timestamp = time.time()
        self.metadata = metadata or {}


class MetricsCollector:
    """
    Thread-safe, in-memory metrics collector.

    Stores operation records per layer and provides aggregate stats.

    Upgrade path:
        - Export to Prometheus / StatsD for Stage 5 dashboards
        - Persist to disk for historical analysis
        - Add anomaly detection thresholds
    """

    def __init__(self):
        self._records: List[_OperationRecord] = []
        self._lock = threading.Lock()

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(
        self,
        layer: str,
        operation: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a completed operation.

        Args:
            layer:     Layer name (e.g., "speech_processing", "storage")
            operation: Operation name (e.g., "asr_transcribe", "embed_chunks")
            duration:  Time in seconds
            success:   Whether the operation succeeded
            error:     Error message if failed
            metadata:  Additional context (e.g., {"chunk_count": 15})
        """
        rec = _OperationRecord(
            layer=layer,
            operation=operation,
            duration=duration,
            success=success,
            error=error,
            metadata=metadata,
        )
        with self._lock:
            self._records.append(rec)

    def timed(self, layer: str, operation: str):
        """
        Context manager for automatic timing + recording.

        Usage:
            with metrics.timed("storage", "save_chunks"):
                store.save_chunks(...)
        """
        return _TimedContext(self, layer, operation)

    # ── Querying ──────────────────────────────────────────────────────────────

    def get_layer_stats(self, layer: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregate stats, optionally filtered by layer.

        Returns:
            {
                "layer_name": {
                    "total_ops": int,
                    "success_count": int,
                    "failure_count": int,
                    "failure_rate": float,
                    "avg_duration_s": float,
                    "max_duration_s": float,
                    "min_duration_s": float,
                    "total_duration_s": float,
                    "operations": {
                        "op_name": {"count": int, "avg_s": float, "failures": int}
                    }
                }
            }
        """
        with self._lock:
            records = list(self._records)

        # Group by layer
        by_layer: Dict[str, List[_OperationRecord]] = defaultdict(list)
        for r in records:
            if layer and r.layer != layer:
                continue
            by_layer[r.layer].append(r)

        stats = {}
        for lyr, recs in sorted(by_layer.items()):
            durations = [r.duration for r in recs]
            successes = [r for r in recs if r.success]
            failures = [r for r in recs if not r.success]

            # Per-operation breakdown
            by_op: Dict[str, List[_OperationRecord]] = defaultdict(list)
            for r in recs:
                by_op[r.operation].append(r)

            op_stats = {}
            for op_name, op_recs in sorted(by_op.items()):
                op_durations = [r.duration for r in op_recs]
                op_stats[op_name] = {
                    "count": len(op_recs),
                    "avg_s": round(sum(op_durations) / len(op_durations), 3),
                    "failures": sum(1 for r in op_recs if not r.success),
                }

            stats[lyr] = {
                "total_ops": len(recs),
                "success_count": len(successes),
                "failure_count": len(failures),
                "failure_rate": round(len(failures) / len(recs), 3) if recs else 0.0,
                "avg_duration_s": round(sum(durations) / len(durations), 3) if durations else 0.0,
                "max_duration_s": round(max(durations), 3) if durations else 0.0,
                "min_duration_s": round(min(durations), 3) if durations else 0.0,
                "total_duration_s": round(sum(durations), 3),
                "operations": op_stats,
            }

        return stats

    def get_recent_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent N errors across all layers."""
        with self._lock:
            errors = [r for r in self._records if not r.success]

        return [
            {
                "layer": r.layer,
                "operation": r.operation,
                "error": r.error,
                "duration_s": round(r.duration, 3),
                "timestamp": r.timestamp,
            }
            for r in errors[-n:]
        ]

    def reset(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._records.clear()


class _TimedContext:
    """Context manager for automatic timing."""

    def __init__(self, collector: MetricsCollector, layer: str, operation: str):
        self._collector = collector
        self._layer = layer
        self._operation = operation
        self._t0 = 0.0
        self._error: Optional[str] = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._t0
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self._collector.record(
            self._layer,
            self._operation,
            duration=duration,
            success=success,
            error=error,
        )
        return False  # Don't suppress exceptions


# ── Global singleton ──────────────────────────────────────────────────────────
# Import anywhere with: from utils.metrics import metrics
metrics = MetricsCollector()
