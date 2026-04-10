"""
layers/storage/store.py — Layer 4: Storage

Responsibility:
    Persist structured meeting data and enable efficient retrieval.

Backends:
    - ChromaDB  — vector embeddings + metadata (semantic search)
    - SQLite    — relational records (meetings, chunks, intelligence)

Architecture principles applied:
    - Separation of concerns: vector ops and relational ops are isolated
    - Plug-and-play: replace either backend without touching other layers
    - Output contract: all public methods accept/return typed Pydantic models
    - Observability: every write operation is logged

Fixes applied (audit):
    - [#4] Transactional writes with rollback on failure (SQLite ↔ ChromaDB sync)
    - [#8] ChromaDB upserts batched in groups of 5000
    - [#9] Added close() method and __del__ for resource cleanup
    - [#13] Added database indexes on foreign keys for query performance
"""

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

import chromadb

import config
from contracts.interfaces import BaseStore
from contracts.schemas import Chunk, IntelligenceOutput, MeetingRecord
from utils.logger import get_layer_logger
from utils.metrics import metrics

logger = get_layer_logger("storage")

# ChromaDB has internal batch limits. Keep upserts below this.
_CHROMA_BATCH_SIZE = 5000


class MeetingStore(BaseStore):
    """
    Dual-backend storage layer.

    ChromaDB  → chunk embeddings + lightweight metadata (fast cosine search)
    SQLite    → full text, meeting records, intelligence outputs (structured queries)

    Upgrade path:
        - Swap ChromaDB for Pinecone / Weaviate / FAISS via the same public API
        - Swap SQLite for PostgreSQL without affecting callers
    """

    def __init__(self):
        self._init_chroma()
        self._init_sqlite()
        logger.info("MeetingStore ready.")

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def _init_chroma(self):
        logger.info(f"Connecting to ChromaDB at '{config.CHROMA_PATH}'...")
        self._chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self._chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{config.CHROMA_COLLECTION}' ready "
            f"({self.collection.count()} existing items)."
        )

    # ── SQLite ────────────────────────────────────────────────────────────────

    def _init_sqlite(self):
        logger.info(f"Connecting to SQLite at '{config.SQLITE_PATH}'...")
        self._conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("SQLite schema ready.")

    def _create_tables(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id   TEXT PRIMARY KEY,
                audio_path   TEXT NOT NULL,
                participants TEXT NOT NULL,   -- JSON array
                language     TEXT NOT NULL,
                duration     REAL NOT NULL,
                created_at   TEXT NOT NULL,
                chunk_count  INTEGER NOT NULL DEFAULT 0,
                summary      TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                meeting_id  TEXT NOT NULL,
                text        TEXT NOT NULL,
                pure_text   TEXT DEFAULT '',
                speakers    TEXT NOT NULL,   -- JSON array
                language    TEXT NOT NULL,
                start_time  REAL NOT NULL,
                end_time    REAL NOT NULL,
                word_count  INTEGER NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
            );

            CREATE TABLE IF NOT EXISTS intelligence (
                meeting_id  TEXT PRIMARY KEY,
                summary     TEXT NOT NULL,
                entities    TEXT NOT NULL,   -- JSON array
                intents     TEXT NOT NULL,   -- JSON array
                FOREIGN KEY (meeting_id) REFERENCES meetings (meeting_id)
            );

            -- Performance indexes for query patterns (fix #13)
            CREATE INDEX IF NOT EXISTS idx_chunks_meeting_id
                ON chunks (meeting_id);
            CREATE INDEX IF NOT EXISTS idx_meetings_created_at
                ON meetings (created_at DESC);
            """
        )
        self._conn.commit()

        # Dynamic Schema Migration: Add pure_text if it doesn't exist
        columns = [row["name"] for row in self._conn.execute("PRAGMA table_info(chunks)").fetchall()]
        if "pure_text" not in columns:
            logger.info("Migrating schema: adding 'pure_text' column to 'chunks' table.")
            self._conn.execute("ALTER TABLE chunks ADD COLUMN pure_text TEXT DEFAULT ''")
            self._conn.commit()

    # ── Public Write API ──────────────────────────────────────────────────────

    def save_meeting(self, record: MeetingRecord) -> None:
        """Persist or update a meeting record in SQLite."""
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO meetings
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.meeting_id,
                    record.audio_path,
                    json.dumps(record.participants),
                    record.language,
                    record.duration,
                    record.created_at,
                    record.chunk_count,
                    record.summary,
                ),
            )
            self._conn.commit()
            logger.info(f"Meeting '{record.meeting_id}' saved to SQLite.")
        except Exception as exc:
            self._conn.rollback()
            logger.error(f"Failed to save meeting '{record.meeting_id}': {exc}")
            raise

    def save_chunks_with_embeddings(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Persist chunks to SQLite (full text) and ChromaDB (embeddings).
        Both stores are kept in sync per chunk_id.

        Fix #4: Transactional — rolls back SQLite if ChromaDB fails.
        Fix #8: ChromaDB upserts batched in groups of 5000.
        """
        if not chunks:
            logger.warning("No chunks to save.")
            return

        t0 = time.time()

        # ── Step 1: SQLite writes (transactional) ─────────────────────────────
        try:
            cursor = self._conn.cursor()
            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.meeting_id,
                        chunk.text,
                        chunk.pure_text,
                        json.dumps(chunk.speakers),
                        chunk.language,
                        chunk.start,
                        chunk.end,
                        chunk.word_count,
                    ),
                )
        except Exception as exc:
            self._conn.rollback()
            logger.error(f"SQLite chunk write failed: {exc}. Rolling back.")
            raise

        # ── Step 2: ChromaDB writes (batched) ─────────────────────────────────
        try:
            for batch_start in range(0, len(chunks), _CHROMA_BATCH_SIZE):
                batch_end = min(batch_start + _CHROMA_BATCH_SIZE, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                batch_embeddings = embeddings[batch_start:batch_end]

                chroma_ids = [c.chunk_id for c in batch_chunks]
                chroma_docs = [c.text for c in batch_chunks]
                chroma_metas = [
                    {
                        "meeting_id": c.meeting_id,
                        "speakers":   ",".join(c.speakers),
                        "language":   c.language,
                        "start":      c.start,
                        "end":        c.end,
                    }
                    for c in batch_chunks
                ]

                self.collection.upsert(
                    ids=chroma_ids,
                    documents=chroma_docs,
                    embeddings=batch_embeddings,
                    metadatas=chroma_metas,
                )

                if batch_end < len(chunks):
                    logger.info(
                        f"ChromaDB batch {batch_start}–{batch_end} of {len(chunks)} upserted."
                    )

        except Exception as exc:
            # ChromaDB failed — rollback SQLite to maintain sync
            self._conn.rollback()
            logger.error(
                f"ChromaDB upsert failed: {exc}. "
                "SQLite changes rolled back to maintain data consistency."
            )
            raise

        # ── Step 3: Commit SQLite only after ChromaDB succeeds ────────────────
        self._conn.commit()

        elapsed = round(time.time() - t0, 3)
        logger.info(
            f"Saved {len(chunks)} chunks to SQLite + ChromaDB in {elapsed}s."
        )

    def save_intelligence(self, output: IntelligenceOutput) -> None:
        """Persist intelligence output and update meeting summary."""
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO intelligence VALUES (?, ?, ?, ?)
                """,
                (
                    output.meeting_id,
                    output.summary,
                    json.dumps(output.entities),
                    json.dumps(output.intents),
                ),
            )
            # Propagate summary back to meeting record
            self._conn.execute(
                "UPDATE meetings SET summary=? WHERE meeting_id=?",
                (output.summary, output.meeting_id),
            )
            self._conn.commit()
            logger.info(
                f"Intelligence saved for meeting '{output.meeting_id}'."
            )
        except Exception as exc:
            self._conn.rollback()
            logger.error(f"Failed to save intelligence for '{output.meeting_id}': {exc}")
            raise

    # ── Public Read API ───────────────────────────────────────────────────────

    def get_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM meetings WHERE meeting_id=?", (meeting_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["participants"] = json.loads(d["participants"])
        return d

    def get_meeting_chunks(self, meeting_id: str) -> List[Dict[str, Any]]:
        """Fetch all chunks for a specific meeting."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE meeting_id=? ORDER BY start_time ASC", (meeting_id,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["speakers"] = json.loads(d["speakers"])
            results.append(d)
        return results

    def list_meetings(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT meeting_id, created_at, chunk_count, summary FROM meetings "
            "ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_intelligence(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM intelligence WHERE meeting_id=?", (meeting_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["entities"] = json.loads(d["entities"])
        d["intents"] = json.loads(d["intents"])
        return d

    def get_all_intelligence(self) -> List[Dict[str, Any]]:
        """
        Stage 3: Fetches intelligence documents for ALL meetings. 
        Used to generate cross-meeting insights.
        """
        rows = self._conn.execute("SELECT * FROM intelligence").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["entities"] = json.loads(d["entities"])
            d["intents"] = json.loads(d["intents"])
            results.append(d)
        return results

    def delete_meeting(self, meeting_id: str) -> bool:
        """
        Delete a meeting and all associated data from both backends.
        Essential for re-ingestion and data cleanup (gap #7).

        Returns True if meeting existed and was deleted.
        """
        try:
            # Check existence
            existing = self.get_meeting(meeting_id)
            if not existing:
                logger.warning(f"Meeting '{meeting_id}' not found for deletion.")
                return False

            # Get chunk IDs for ChromaDB cleanup
            chunk_rows = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE meeting_id=?", (meeting_id,)
            ).fetchall()
            chunk_ids = [row["chunk_id"] for row in chunk_rows]

            # Delete from SQLite
            self._conn.execute("DELETE FROM intelligence WHERE meeting_id=?", (meeting_id,))
            self._conn.execute("DELETE FROM chunks WHERE meeting_id=?", (meeting_id,))
            self._conn.execute("DELETE FROM meetings WHERE meeting_id=?", (meeting_id,))
            self._conn.commit()

            # Delete from ChromaDB
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)

            # Delete artifact file
            artifact_path = config.MEETINGS_DIR / f"{meeting_id}.json"
            if artifact_path.exists():
                artifact_path.unlink()
                logger.info(f"Artifact file deleted: {artifact_path}")

            logger.info(
                f"Meeting '{meeting_id}' deleted — "
                f"{len(chunk_ids)} chunks removed from SQLite + ChromaDB."
            )
            return True

        except Exception as exc:
            self._conn.rollback()
            logger.error(f"Failed to delete meeting '{meeting_id}': {exc}")
            raise

    # ── Resource Management (fix #9) ──────────────────────────────────────────

    def close(self):
        """Explicitly close database connections. Call when done."""
        if self._conn:
            try:
                self._conn.close()
                logger.info("SQLite connection closed.")
            except Exception as exc:
                logger.warning(f"SQLite close error: {exc}")
        self._conn = None

    def __del__(self):
        """Ensure connections are cleaned up on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
