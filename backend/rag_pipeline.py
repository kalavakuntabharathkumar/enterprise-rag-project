import time

from langchain_core.documents import Document

from analytics.query_log import QueryAnalytics
from backend.config import Config
from backend.inference_client import InferenceError, generate as llm_generate
from backend.latency import record_generation_time, record_retrieval_time
from backend.logger import app_logger, query_logger
from backend.retriever import DocumentRetriever
from backend.utils import extract_text_from_pdf
from ml.intent_classifier import IntentClassifier
from ml.reranker import LexicalReranker
from models.intent_classifier import QueryTypeClassifier
from preprocessing.pipeline import PreprocessingPipeline

CHARS_PER_TOKEN = 4  # rough estimate; good enough for a cost/usage proxy, not billing

CANNED_RESPONSES = {
    "greeting": "Hello! Upload a PDF and ask me a question about it whenever you're ready.",
    "chit_chat": "I'm a document assistant focused on the PDFs you upload — ask me something about their content.",
}


class RAGPipeline:
    def __init__(self):
        # `retriever` touches OpenAI credentials lazily (see DocumentRetriever's
        # own lazy embeddings), so RAGPipeline — and therefore the FastAPI app —
        # can be constructed at import time even when OPENAI_API_KEY isn't set
        # yet.  This is what lets /health report openai_configured=false instead
        # of the process failing to boot.
        self.retriever = DocumentRetriever()
        self.preprocessor = PreprocessingPipeline(min_quality=Config.MIN_CHUNK_QUALITY)
        self.intent_classifier = IntentClassifier()
        self.query_type_classifier = QueryTypeClassifier()
        self.reranker = LexicalReranker()
        self.analytics = QueryAnalytics()

    def process_pdf(self, file_path: str):
        """Process PDF: extract, clean, chunk, dedupe, score, embed, store."""
        try:
            text = extract_text_from_pdf(file_path)
            records = self.preprocessor.run(text, source=file_path)

            documents = [
                Document(page_content=r["text"], metadata=r["metadata"])
                for r in records
            ]

            if not documents:
                app_logger.warning(f"No indexable chunks survived preprocessing for {file_path}")
                return {
                    "status": "error",
                    "message": "No usable text could be extracted from this PDF after cleaning and quality filtering.",
                    "chunks_indexed": 0,
                }

            self.retriever.add_documents(documents)

            app_logger.info(f"Processed PDF: {file_path} ({len(documents)} chunks indexed)")
            return {
                "status": "success",
                "message": "PDF processed successfully",
                "chunks_indexed": len(documents),
            }
        except Exception as e:
            app_logger.error(f"Error processing PDF: {e}")
            return {"status": "error", "message": str(e), "chunks_indexed": 0}

    def ask_question(self, question: str):
        """Answer question using RAG, with intent routing and lexical
        re-ranking on top of embedding retrieval."""
        start = time.time()
        intent = self.intent_classifier.predict(question)

        if intent != "document_question":
            latency_ms = (time.time() - start) * 1000
            self._log_query(question, intent, None, [], 0.0, 0.0, 1.0, latency_ms,
                             answered=True, llm_skipped=True, estimated_tokens=0,
                             estimated_cost_usd=0.0, llm_backend="none")
            return {
                "answer": CANNED_RESPONSES.get(intent, CANNED_RESPONSES["chit_chat"]),
                "sources": [],
                "confidence": 1.0,
                "intent": intent,
            }

        query_type = self.query_type_classifier.predict(question)

        try:
            # Over-fetch candidates so the lexical re-ranker has room to
            # promote strong keyword matches that embedding similarity
            # alone might rank lower.
            with_timing_start = time.time()
            candidates = self.retriever.retrieve_with_scores(question, k=Config.TOP_K * 3)
            record_retrieval_time((time.time() - with_timing_start) * 1000)

            if not candidates:
                latency_ms = (time.time() - start) * 1000
                self._log_query(question, intent, query_type, [], 0.0, 0.0, 0.0, latency_ms,
                                 answered=False, llm_skipped=True, estimated_tokens=0,
                                 estimated_cost_usd=0.0, llm_backend="none")
                return {
                    "answer": "No relevant information found in the documents.",
                    "sources": [],
                    "confidence": 0.0,
                    "intent": intent,
                    "query_type": query_type,
                }

            doc_texts = [doc.page_content for doc, _ in candidates]
            embedding_scores = [score for _, score in candidates]
            order = self.reranker.rerank(question, doc_texts, embedding_scores, alpha=Config.RERANK_ALPHA)

            top_indices = order[:Config.TOP_K]
            selected = [candidates[i] for i in top_indices]
            docs = [doc for doc, _ in selected]
            similarities = [score for _, score in selected]

            context = "\n".join([doc.page_content for doc in docs])
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            top_similarity = max(similarities) if similarities else 0.0
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            # Factual questions that retrieval already answers with high
            # confidence skip the LLM entirely: composing a sentence
            # around one clearly-retrieved fact doesn't need generation,
            # and it saves the latency of a completion call.
            if query_type == "factual" and top_similarity >= Config.FACTUAL_DIRECT_THRESHOLD:
                record_generation_time(0.0)
                answer = docs[0].page_content.strip()
                latency_ms = (time.time() - start) * 1000
                self._log_query(question, intent, query_type, docs, top_similarity, avg_similarity,
                                 top_similarity, latency_ms, answered=True, llm_skipped=True,
                                 estimated_tokens=0, estimated_cost_usd=0.0, llm_backend="none")
                return {
                    "answer": answer,
                    "sources": sources,
                    "confidence": top_similarity,
                    "intent": intent,
                    "query_type": query_type,
                }

            prompt = f"""Based on the following context, answer the question accurately. If the context doesn't contain enough information to answer confidently, say so.

Context:
{context}

Question: {question}

Answer concisely and cite sources if possible."""

            generation_start = time.time()
            answer = llm_generate(prompt)
            record_generation_time((time.time() - generation_start) * 1000)

            confidence = self._estimate_confidence(answer, context)

            query_logger.info(f"Question: {question} | Answer: {answer} | Confidence: {confidence}")

            if confidence < Config.CONFIDENCE_THRESHOLD:
                answer = "I'm not confident in this answer based on the available information. Please rephrase or provide more context."

            latency_ms = (time.time() - start) * 1000
            estimated_tokens = (len(prompt) + len(answer)) // CHARS_PER_TOKEN
            # Self-hosted inference carries no per-token API cost; set to 0.0.
            estimated_cost_usd = 0.0
            self._log_query(question, intent, query_type, docs, top_similarity, avg_similarity,
                             confidence, latency_ms, answered=True, llm_skipped=False,
                             estimated_tokens=estimated_tokens, estimated_cost_usd=estimated_cost_usd,
                             llm_backend=Config.LLM_BACKEND)

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "intent": intent,
                "query_type": query_type,
            }
        except InferenceError:
            raise
        except Exception as e:
            app_logger.error(f"Error answering question: {e}")
            latency_ms = (time.time() - start) * 1000
            self._log_query(question, intent, query_type, [], 0.0, 0.0, 0.0, latency_ms,
                             answered=False, llm_skipped=True, estimated_tokens=0,
                             estimated_cost_usd=0.0, llm_backend="error")
            return {
                "answer": "An error occurred while processing your question.",
                "sources": [],
                "confidence": 0.0,
                "intent": intent,
                "query_type": query_type,
            }

    def _log_query(self, question, intent, query_type, docs, top_similarity, avg_similarity,
                    confidence, latency_ms, answered, llm_skipped, estimated_tokens,
                    estimated_cost_usd, llm_backend="unknown"):
        try:
            self.analytics.log({
                "timestamp": time.time(),
                "question": question,
                "intent": intent,
                "query_type": query_type,
                "num_docs_retrieved": len(docs),
                "top_similarity": top_similarity,
                "avg_similarity": avg_similarity,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "answered": answered,
                "llm_skipped": llm_skipped,
                "estimated_tokens": estimated_tokens,
                "estimated_cost_usd": estimated_cost_usd,
                "llm_backend": llm_backend,
            })
        except Exception as e:
            app_logger.warning(f"Failed to log query analytics: {e}")

    def _estimate_confidence(self, answer: str, context: str) -> float:
        """Simple confidence estimation based on answer/context word overlap."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(answer_words.intersection(context_words))
        total_words = len(answer_words)
        if total_words == 0:
            return 0.0
        return min(overlap / total_words, 1.0)
