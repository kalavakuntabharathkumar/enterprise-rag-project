import time
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from backend.config import Config
from backend.logger import app_logger
from backend.query_cache import retrieval_cache
import os

class DocumentRetriever:
    def __init__(self):
        # Embeddings are created lazily (see `embeddings` property) so that
        # constructing a DocumentRetriever - and therefore the whole
        # RAGPipeline / FastAPI app - does not require OPENAI_API_KEY to be
        # set at import/startup time. The key is only needed once a
        # document is actually processed or a question is actually asked,
        # which is also when /health can meaningfully report it as missing.
        self._embeddings = None
        self.vectorstore_path = Config.VECTORSTORE_PATH
        self.vectorstore = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=Config.OPENAI_API_KEY, model=Config.EMBEDDING_MODEL
            )
        return self._embeddings

    def load_vectorstore(self):
        """Load existing vectorstore if available.

        `allow_dangerous_deserialization=True` is required by LangChain's
        FAISS integration because its docstore is pickle-based; there is no
        safe-deserialization alternative for local FAISS today. This is
        only safe because `vectorstore_path` is local application storage
        that the API never lets a client write to directly (uploads only
        ever go through PDF extraction) - do not point it at a location
        writable by untrusted parties.
        """
        try:
            if os.path.exists(os.path.join(self.vectorstore_path, "index.faiss")):
                self.vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                app_logger.info("Loaded existing vectorstore")
            else:
                app_logger.info("No existing vectorstore found")
        except Exception as e:
            app_logger.error(f"Error loading vectorstore: {e}")
            self.vectorstore = None

    def save_vectorstore(self):
        """Save vectorstore to disk."""
        if self.vectorstore:
            os.makedirs(self.vectorstore_path, exist_ok=True)
            self.vectorstore.save_local(self.vectorstore_path)
            app_logger.info("Saved vectorstore to disk")

    def _embed_in_batches(self, documents: List[Document], batch_size: int) -> List[list]:
        """Embed document page_content in batches of `batch_size`, returning
        a flat list of embedding vectors aligned 1-to-1 with `documents`.

        Sending chunks to the OpenAI embedding API in controlled batches
        (rather than all at once) gives explicit control over request size,
        keeps individual API calls within token-rate-limit budgets, and
        makes ingestion progress visible in logs. Each batch is a single
        HTTP request; the total number of requests is ceil(n / batch_size).
        """
        texts = [doc.page_content for doc in documents]
        n = len(texts)
        all_embeddings: List[list] = []
        ingestion_start = time.time()

        app_logger.info(
            f"Embedding {n} chunks in batches of {batch_size} "
            f"({-(-n // batch_size)} API call(s))"
        )

        for batch_start in range(0, n, batch_size):
            batch_texts = texts[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            batch_t0 = time.time()

            batch_vectors = self.embeddings.embed_documents(batch_texts)
            batch_ms = (time.time() - batch_t0) * 1000

            all_embeddings.extend(batch_vectors)
            app_logger.info(
                f"  batch {batch_num}: {len(batch_texts)} chunks embedded "
                f"in {batch_ms:.0f} ms"
            )

        total_ms = (time.time() - ingestion_start) * 1000
        app_logger.info(
            f"Ingestion embedding complete: {n} chunks in {total_ms:.0f} ms "
            f"({total_ms / n:.1f} ms/chunk avg)"
        )
        return all_embeddings

    def add_documents(self, documents: List[Document], metadatas=None):
        """Embed `documents` in explicit batches then add pre-computed
        vectors to the FAISS index.

        Using pre-computed embeddings (via FAISS.from_embeddings /
        vectorstore.add_embeddings) rather than passing raw documents to
        FAISS.from_documents gives us:
        - Explicit batch-size control over embedding API requests
        - Per-batch timing visible in the application log
        - A single place to swap the embedding backend without touching
          FAISS construction logic

        No-op on an empty list rather than letting FAISS raise on an
        empty corpus.
        """
        if not documents:
            app_logger.warning("add_documents called with no documents; skipping")
            return

        if self.vectorstore is None:
            self.load_vectorstore()

        vectors = self._embed_in_batches(documents, batch_size=Config.EMBED_BATCH_SIZE)

        # Build (text, embedding) pairs; keep metadatas aligned.
        doc_metadatas = [doc.metadata for doc in documents]
        texts = [doc.page_content for doc in documents]
        text_embedding_pairs = list(zip(texts, vectors))

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_embeddings(
                text_embedding_pairs,
                self.embeddings,
                metadatas=doc_metadatas,
            )
        else:
            self.vectorstore.add_embeddings(
                text_embedding_pairs,
                metadatas=doc_metadatas,
            )

        self.save_vectorstore()
        app_logger.info(f"Added {len(documents)} documents to vectorstore")

    def retrieve(self, query: str, k: int = Config.TOP_K):
        """Retrieve top-k similar documents."""
        return [doc for doc, _ in self.retrieve_with_scores(query, k=k)]

    def retrieve_with_scores(self, query: str, k: int = Config.TOP_K):
        """Retrieve top-k documents along with a normalized similarity
        score in (0, 1], where 1 means an exact match. FAISS returns an L2
        distance (lower is more similar); we convert it to a similarity so
        it can be combined with lexical re-ranking scores and logged
        alongside confidence in the analytics layer.

        Results are cached in-memory by (query, k) so that an identical
        query string never triggers a second OpenAI embedding call within
        the same process lifetime. See backend/query_cache.py for details
        and limitations.
        """
        hit, cached = retrieval_cache.get(query, k)
        if hit:
            app_logger.info(f"Cache hit for query (k={k}): {query}")
            return cached

        if self.vectorstore is None:
            self.load_vectorstore()
        if self.vectorstore is None:
            return []
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            scored = [(doc, 1.0 / (1.0 + score)) for doc, score in results]
            app_logger.info(f"Retrieved {len(scored)} documents for query: {query}")
            retrieval_cache.set(query, k, scored)
            return scored
        except Exception as e:
            app_logger.error(f"Error retrieving documents: {e}")
            return []

    def get_stats(self) -> dict:
        """Real index stats for the /stats endpoint: unique source
        documents and total chunks pulled from the live FAISS docstore
        (not tracked separately, so this can never drift from the actual
        index), plus the on-disk size of the persisted index files."""
        if self.vectorstore is None:
            self.load_vectorstore()

        total_chunks = 0
        sources = set()
        if self.vectorstore is not None:
            try:
                total_chunks = int(self.vectorstore.index.ntotal)
            except Exception as e:
                app_logger.warning(f"Could not read FAISS index size: {e}")
            try:
                for doc in self.vectorstore.docstore._dict.values():
                    sources.add(doc.metadata.get("source", "unknown"))
            except Exception as e:
                app_logger.warning(f"Could not enumerate vectorstore docstore: {e}")

        size_bytes = 0
        for filename in ("index.faiss", "index.pkl"):
            file_path = os.path.join(self.vectorstore_path, filename)
            if os.path.exists(file_path):
                size_bytes += os.path.getsize(file_path)

        return {
            "total_documents": len(sources),
            "total_chunks": total_chunks,
            "vector_db_size_bytes": size_bytes,
        }