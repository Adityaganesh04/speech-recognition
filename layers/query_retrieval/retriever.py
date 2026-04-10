import time
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from contracts.interfaces import BaseRetriever
from contracts.schemas import QueryInput, RetrievedChunk
from utils.logger import get_layer_logger

logger = get_layer_logger("query_retrieval")


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever: Combines BM25 (Keyword) and Dense (Semantic) search.
    Uses Reciprocal Rank Fusion (RRF) for production-grade results.
    """

    def __init__(self, chroma_collection):
        logger.info(f"Loading embedding model: '{config.EMBEDDING_MODEL}'...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.collection = chroma_collection

        # In-memory index state
        self.bm25 = None
        self.documents = []
        self.metadatas = []

        logger.info("HybridRetriever initialized (BM25 + Dense).")

    # ─────────────────────────────────────────────
    # Build BM25 index
    # ─────────────────────────────────────────────
    def _build_bm25_index(self, meeting_id: Optional[str] = None):
        where = {"meeting_id": meeting_id} if meeting_id else None

        results = self.collection.get(
            where=where,
            include=["documents", "metadatas"]
        )

        if not results["documents"]:
            logger.warning("No documents found to build BM25 index.")
            return False

        self.documents = results["documents"]
        self.metadatas = results["metadatas"]

        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        return True

    # ─────────────────────────────────────────────
    # Retrieve
    # ─────────────────────────────────────────────
    def retrieve(self, query_input: QueryInput) -> List[RetrievedChunk]:
        t0 = time.time()

        # 1. Build BM25 index
        if not self._build_bm25_index(query_input.meeting_id):
            return []

        # 2. Dense Search
        query_embedding = self.embedder.encode(
            query_input.query,
            normalize_embeddings=True
        ).tolist()

        where = {"meeting_id": query_input.meeting_id} if query_input.meeting_id else None

        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(query_input.top_k * 2, len(self.documents)),
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # 3. BM25 Search
        query_tokens = query_input.query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranking_idx = np.argsort(bm25_scores)[::-1][:query_input.top_k * 2].tolist()

        # 4. Reciprocal Rank Fusion (RRF)
        k = 60
        rrf_scores = {}

        # Dense ranking
        for rank, doc_id in enumerate(dense_results["ids"][0]):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank + 1))

        # BM25 ranking (fixed mapping)
        for rank, idx in enumerate(bm25_ranking_idx):
            metadata = self.metadatas[idx]
            doc_id = metadata.get("chunk_id")

            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank + 1))

        # 5. Final sorting
        sorted_ids = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:query_input.top_k]

        results = []

        if sorted_ids:
            final_data = self.collection.get(
                ids=[sid[0] for sid in sorted_ids],
                include=["documents", "metadatas"]
            )

            for i, doc_id in enumerate(final_data["ids"]):
                meta = final_data["metadatas"][i]

                # FIX: ensure speakers is always a list
                raw_speakers = meta.get("speakers", [])
                if isinstance(raw_speakers, str):
                    speakers = [raw_speakers]
                else:
                    speakers = raw_speakers

                results.append(
                    RetrievedChunk(
                        chunk_id=doc_id,
                        text=final_data["documents"][i],
                        score=round(rrf_scores[doc_id], 4),
                        speakers=speakers,
                        meeting_id=meta.get("meeting_id", "")
                    )
                )

        elapsed = round(time.time() - t0, 3)
        logger.info(f"Hybrid retrieval (BM25 + Dense) complete in {elapsed}s.")

        return results

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        return [
            e.tolist()
            for e in self.embedder.encode(texts, normalize_embeddings=True)
        ]