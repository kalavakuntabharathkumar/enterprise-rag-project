"""Smoke tests confirming FAISS retrieval, TF-IDF intent routing, and
lexical re-ranking are intact after the switch to a self-hosted generation
backend.

These tests exercise component instantiation and basic contract — they do
not require a running model server, a populated vector index, or an
OpenAI API key.  Run with:

    python -m pytest tests/test_retrieval_stack.py -v
"""
import pytest

from ml.intent_classifier import IntentClassifier
from ml.reranker import LexicalReranker
from backend.retriever import DocumentRetriever


# ---------------------------------------------------------------------------
# Intent classifier (TF-IDF + Logistic Regression routing)
# ---------------------------------------------------------------------------

class TestIntentClassifier:
    def setup_method(self):
        self.clf = IntentClassifier()

    def test_greeting_routed_correctly(self):
        intent = self.clf.predict("Hello, how are you?")
        assert intent in {"greeting", "chit_chat"}, (
            f"Expected greeting/chit_chat for a plain greeting, got '{intent}'"
        )

    def test_document_question_routed_correctly(self):
        intent = self.clf.predict("What are the main findings of the report?")
        assert intent == "document_question", (
            f"Expected document_question for a content query, got '{intent}'"
        )

    def test_predict_returns_string(self):
        result = self.clf.predict("Summarise the methodology section.")
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# Lexical re-ranker (TF-IDF cosine + embedding score blend)
# ---------------------------------------------------------------------------

class TestLexicalReranker:
    def setup_method(self):
        self.reranker = LexicalReranker()

    def test_rerank_returns_full_index_list(self):
        docs = ["The cat sat on the mat.", "Neural networks learn representations.", "Paris is the capital of France."]
        scores = [0.8, 0.6, 0.7]
        order = self.reranker.rerank("cat mat", docs, scores, alpha=0.5)
        assert sorted(order) == list(range(len(docs))), (
            "rerank must return a permutation of all input indices"
        )

    def test_rerank_promotes_keyword_match(self):
        docs = ["Photosynthesis converts sunlight to energy.", "The cat sat on the mat."]
        # Give the keyword-weak doc a higher embedding score; the re-ranker
        # should still prefer the one whose text matches the query terms.
        scores = [0.9, 0.5]
        order = self.reranker.rerank("cat mat", docs, scores, alpha=0.0)
        # alpha=0 means pure lexical — the second doc should rank first
        assert order[0] == 1, "Pure-lexical re-rank should prefer the keyword-matching doc"

    def test_rerank_single_doc(self):
        order = self.reranker.rerank("anything", ["Only one document."], [0.5], alpha=0.5)
        assert order == [0]


# ---------------------------------------------------------------------------
# DocumentRetriever (FAISS)
# ---------------------------------------------------------------------------

class TestDocumentRetriever:
    def test_instantiation_does_not_require_credentials(self):
        """DocumentRetriever must construct without an OPENAI_API_KEY set,
        because embeddings are created lazily."""
        retriever = DocumentRetriever()
        assert retriever is not None
        # vectorstore starts as None — no index on disk in test environment
        assert retriever.vectorstore is None

    def test_retrieve_with_empty_store_returns_empty_list(self):
        retriever = DocumentRetriever()
        # No vectorstore loaded — must return [] rather than raise
        results = retriever.retrieve_with_scores("any query", k=3)
        assert results == [], (
            "retrieve_with_scores on an empty store should return [], not raise"
        )

    def test_get_stats_with_empty_store(self):
        retriever = DocumentRetriever()
        stats = retriever.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0
        assert stats["vector_db_size_bytes"] == 0
