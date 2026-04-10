"""
layers/query_retrieval/retriever.py — Layer 5: Query & Retrieval

Responsibility:
    Embed user queries and fetch semantically similar chunks from storage.

Pipeline:
    QueryInput → Query Embedding → ChromaDB Similarity Search → List[RetrievedChunk]

Architecture principles applied:
    - Plug-and-play: embedding model swappable via config.EMBEDDING_MODEL
    - Output contract: always returns List[RetrievedChunk]
    - Shared resource: embedder is also used by the pipeline to embed stored chunks
    - Observability: retrieval count, threshold filtering, and timing are logged
"""

import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

import config
from contracts.interfaces import BaseRetriever
from contracts.schemas import QueryInput, RetrievedChunk
from utils.logger import get_layer_logger
from utils.metrics import metrics

logger = get_layer_logger("query_retrieval")


class SemanticRetriever(BaseRetriever):
    """
    Embedding-based semantic retrieval over ChromaDB.

    Upgrade path:
        - Swap SentenceTransformer for OpenAI embeddings, Cohere, etc. via config
        - Add hybrid (BM25 + dense) search in Stage 2
        - Add re-ranking (cross-encoder) pass after initial retrieval
    """

    def __init__(self, chroma_collection):
        logger.info(f"Loading embedding model: '{config.EMBEDDING_MODEL}'...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.collection = chroma_collection
        self._validate_embedding_model()
        logger.info("SemanticRetriever ready.")

    def _validate_embedding_model(self):
        """
        Check if the configured embedding model matches what's stored in ChromaDB.
        Prevents silent data corruption from model mismatches (gap #4).
        """
        try:
            meta = self.collection.metadata or {}
            stored_model = meta.get("embedding_model")

            if stored_model is None:
                # First time — record the model
                self.collection.modify(
                    metadata={
                        **meta,
                        "hnsw:space": "cosine",
                        "embedding_model": config.EMBEDDING_MODEL,
                    }
                )
                logger.info(f"Embedding model '{config.EMBEDDING_MODEL}' registered in ChromaDB.")
            elif stored_model != config.EMBEDDING_MODEL:
                logger.warning(
                    f"⚠ EMBEDDING MODEL MISMATCH: "
                    f"ChromaDB has embeddings from '{stored_model}' "
                    f"but config specifies '{config.EMBEDDING_MODEL}'. "
                    f"Query results may be inaccurate. "
                    f"Re-ingest all meetings or revert EMBEDDING_MODEL to '{stored_model}'."
                )
        except Exception as exc:
            logger.warning(f"Could not validate embedding model: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query_input: QueryInput) -> List[RetrievedChunk]:
        """
        Embed query and fetch top-k semantically similar chunks.

        Args:
            query_input: QueryInput contract (query text, optional meeting scope, top_k).

        Returns:
            List[RetrievedChunk] sorted by descending similarity score.
        """
        t0 = time.time()
        logger.info(
            f"Querying: '{query_input.query[:70]}' "
            f"(top_k={query_input.top_k}, "
            f"meeting_filter={query_input.meeting_id or 'all'})"
        )

        total_items = self.collection.count()
        if total_items == 0:
            logger.warning("ChromaDB collection is empty — no meetings ingested yet.")
            return []

        # Embed the query
        query_embedding = self.embedder.encode(
            query_input.query, normalize_embeddings=True
        ).tolist()

        # Build optional metadata filter
        where: Optional[dict] = None
        if query_input.meeting_id:
            where = {"meeting_id": query_input.meeting_id}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(query_input.top_k, total_items),
            where=where,
            include=["documents", "distances", "metadatas"],
        )

        # Parse and filter results
        retrieved: List[RetrievedChunk] = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            score = round(1.0 - distance, 4)   # ChromaDB cosine: distance = 1 - similarity

            if score < config.SIMILARITY_THRESHOLD:
                logger.debug(
                    f"Chunk '{results['ids'][0][i]}' filtered out "
                    f"(score={score} < threshold={config.SIMILARITY_THRESHOLD})."
                )
                continue

            meta = results["metadatas"][0][i]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=results["ids"][0][i],
                    text=doc,
                    score=score,
                    speakers=meta.get("speakers", "SPEAKER_00").split(","),
                    meeting_id=meta.get("meeting_id", ""),
                )
            )

        # Sort by descending score
        retrieved.sort(key=lambda x: x.score, reverse=True)

        elapsed = round(time.time() - t0, 3)
        logger.info(
            f"Retrieved {len(retrieved)} chunks "
            f"(above threshold={config.SIMILARITY_THRESHOLD}) in {elapsed}s."
        )
        return retrieved

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """
        Batch-embed a list of chunk texts for storage.
        Shared between the pipeline (ingestion) and query paths.
        """
        logger.info(f"Embedding {len(texts)} chunks...")
        t0 = time.time()
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed = round(time.time() - t0, 3)
        logger.info(f"Chunk embedding complete in {elapsed}s.")
        return [e.tolist() for e in embeddings]
