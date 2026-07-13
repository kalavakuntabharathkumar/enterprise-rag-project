import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K = int(os.getenv("TOP_K", 3))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MIN_CHUNK_QUALITY = float(os.getenv("MIN_CHUNK_QUALITY", 0.35))
    RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", 0.6))
    # Number of chunks sent to the embedding API in a single call during
    # document ingestion. Larger values reduce API round-trips; smaller
    # values reduce peak memory and make progress logging finer-grained.
    # OpenAI's text-embedding-3-* models accept up to 2048 inputs per call,
    # so any value up to that is valid. 32 is a conservative default that
    # keeps individual requests well within rate-limit token budgets.
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 32))
    # Stub auth: unset means the API is open. Set API_KEY to require callers
    # to send a matching X-API-Key header on /upload and /ask.
    API_KEY = os.getenv("API_KEY")
    # Factual questions whose top retrieved chunk clears this similarity
    # bar are answered by returning that chunk directly, skipping the LLM
    # call entirely. Summarization/comparison questions always go through
    # the LLM since they need synthesis across chunks.
    FACTUAL_DIRECT_THRESHOLD = float(os.getenv("FACTUAL_DIRECT_THRESHOLD", 0.82))
    # Rough per-1K-token cost estimates (USD) used only for the /stats
    # endpoint's estimated_cost_per_query figure.
    LLM_INPUT_COST_PER_1K = float(os.getenv("LLM_INPUT_COST_PER_1K", 0.0005))
    LLM_OUTPUT_COST_PER_1K = float(os.getenv("LLM_OUTPUT_COST_PER_1K", 0.0015))

    # Self-hosted inference backend.
    # LLM_BACKEND selects which server type is running.  Accepted values:
    #   vllm      — vLLM; default base URL http://llm-server:8000
    #   llamacpp  — llama.cpp server; default base URL http://llm-server:8080
    # Set LLM_BASE_URL to override the default for either backend.
    LLM_BACKEND = os.getenv("LLM_BACKEND", "vllm")

    # Default base URLs per backend (no trailing slash, no /v1 suffix).
    _BACKEND_DEFAULTS = {
        "vllm": "http://llm-server:8000",
        "llamacpp": "http://llm-server:8080",
    }

    @classmethod
    def get_llm_base_url(cls) -> str:
        """Return the effective base URL: explicit LLM_BASE_URL takes
        priority; otherwise fall back to the per-backend default."""
        explicit = os.getenv("LLM_BASE_URL")
        if explicit:
            return explicit.rstrip("/")
        return cls._BACKEND_DEFAULTS.get(cls.LLM_BACKEND, "http://llm-server:8000")

    # LLM_MODEL: model name sent in the request body.  For vLLM it must
    # match the --model argument the server was started with; for llama.cpp
    # it may be any non-empty string (the server ignores it).
    LLM_MODEL = os.getenv("LLM_MODEL", "mistral-7b-instruct")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
    # Seconds before the inference call is treated as failed.
    LLM_TIMEOUT_SECS = float(os.getenv("LLM_TIMEOUT_SECS", 120.0))
