import json
import logging
import os
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder
from backend.config import GEMINI_API_KEY
from backend.vector_store import vector_store
from backend.cache import query_cache

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging - both console and file
logger = logging.getLogger("RAGService")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:
    file_handler = logging.FileHandler("logs/rag_service.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator for retrying functions with exponential backoff.
    
    Why Retry Logic?
    - LLM APIs can have transient failures
    - Network issues may cause temporary errors
    - Rate limiting requires backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"[RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"[RETRY] All {max_retries} attempts failed: {e}")
            raise last_exception
        return wrapper
    return decorator


class RAGService:
    """
    Production-Ready RAG Service with:
    - Conversational query rewriting
    - Query decomposition
    - Parallel retrieval
    - Cross-encoder reranking
    - LRU caching
    - Retry with exponential backoff
    """
    
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = "gemini-2.5-flash"
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("RAGService initialized with caching and retry support")

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_llm(self, prompt: str, temperature: float = 0.1, system_instruction: str = None) -> str:
        """Call LLM with retry logic."""
        config = types.GenerateContentConfig(temperature=temperature)
        if system_instruction:
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text

    def rewrite_query(self, query: str, history: list[dict]) -> str:
        """Rewrite query to be standalone based on conversation history."""
        if not history:
            logger.info(f"[REWRITE] No history, using original: '{query[:50]}...'")
            return query
        
        logger.info(f"[REWRITE] Processing query with {len(history)} history messages")
        
        recent_history = history[-6:]
        conversation = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:200]}"
            for msg in recent_history
        ])
        
        prompt = f"""
        Rewrite the user's query to be standalone and self-contained based on the conversation history.
        
        Rules:
        1. If the query references previous context (e.g., "it", "that", "the same"), rewrite it to include the referenced information
        2. If the query is already standalone or unrelated to the conversation, return it as-is
        3. Keep the rewritten query concise and natural
        4. Return ONLY the rewritten query, nothing else
        
        Conversation History:
        {conversation}
        
        Current Query: "{query}"
        
        Rewritten Query:"""

        try:
            rewritten = self._call_llm(prompt, temperature=0.1).strip().strip('"').strip("'")
            if rewritten:
                logger.info(f"[REWRITE] '{query[:30]}...' -> '{rewritten[:50]}...'")
                return rewritten
        except Exception as e:
            logger.error(f"[REWRITE] Failed: {e}")
        
        return query

    def decompose_query(self, query: str) -> list[str]:
        """Use LLM to split complex query into sub-queries."""
        logger.info(f"[DECOMPOSE] Analyzing: '{query[:50]}...'")
        
        prompt = f"""
        Analyze this query and decide if it should be split into simpler sub-queries for better document retrieval.

        # Query: "{query}"

        Rules:
        1. If the query is simple and focused, return it as-is (single query)
        2. If the query has multiple aspects or is complex, split into 2-4 sub-queries
        3. Each sub-query should be self-contained and searchable
        4. Return ONLY a JSON array of strings, nothing else

        # Examples:
        - "What is machine learning?" -> ["What is machine learning?"]
        - "Compare Python and Java for web dev" -> ["Python for web development", "Java for web development"]

        Return JSON array:
        ```json
        ["sub_query_1", ...]
        ```
        """

        try:
            result = self._call_llm(prompt, temperature=0.1)
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            
            sub_queries = json.loads(result)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                logger.info(f"[DECOMPOSE] Split into {len(sub_queries)} sub-queries")
                return sub_queries[:4]
        except Exception as e:
            logger.error(f"[DECOMPOSE] Failed: {e}")
        
        return [query]

    def retrieve_for_query(self, query: str, n_results: int = 10) -> tuple[list[str], list[dict]]:
        """Retrieve documents for a single query."""
        results = vector_store.query(query_text=query, n_results=n_results)
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        return documents, metadatas

    def parallel_retrieve(self, queries: list[str]) -> tuple[list[str], list[dict]]:
        """Retrieve documents for multiple queries in parallel."""
        logger.info(f"[RETRIEVE] Parallel retrieval for {len(queries)} queries")
        
        all_docs, all_metas, seen_docs = [], [], set()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.retrieve_for_query, q, 10) for q in queries]
            for future in futures:
                docs, metas = future.result()
                for doc, meta in zip(docs, metas):
                    doc_hash = hash(doc[:200])
                    if doc_hash not in seen_docs:
                        seen_docs.add(doc_hash)
                        all_docs.append(doc)
                        all_metas.append(meta)
        
        logger.info(f"[RETRIEVE] Total unique documents: {len(all_docs)}")
        return all_docs, all_metas

    def rerank(self, query: str, documents: list[str], metadatas: list[dict], top_k: int = 5) -> tuple[list[str], list[dict]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return [], []
        
        logger.info(f"[RERANK] Reranking {len(documents)} docs, selecting top {top_k}")
        
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        doc_meta_scores = sorted(zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True)
        
        top_docs = [d for d, m, s in doc_meta_scores[:top_k]]
        top_metas = [m for d, m, s in doc_meta_scores[:top_k]]
        
        top_scores = [f"{s:.3f}" for d, m, s in doc_meta_scores[:top_k]]
        logger.info(f"[RERANK] Top scores: {top_scores}")
        
        return top_docs, top_metas

    def generate_response(self, query: str, history: list[dict] = None) -> dict:
        """
        Full Conversational RAG pipeline with caching:
        1. Check cache for recent identical query
        2. Rewrite query based on conversation history
        3. Decompose query into sub-queries
        4. Parallel retrieval for all sub-queries
        5. Rerank chunks with rewritten query
        6. Generate answer with context
        7. Cache the result
        """
        history = history or []
        
        logger.info("=" * 60)
        logger.info(f"[PIPELINE] New query: '{query[:80]}...'")
        logger.info(f"[PIPELINE] History messages: {len(history)}")
        
        # 1. Check cache
        cached = query_cache.get(query, len(history))
        if cached:
            logger.info("[CACHE] Cache hit! Returning cached response")
            return cached
        
        # 2. Rewrite query for context
        rewritten_query = self.rewrite_query(query, history)
        
        # 3. Query Decomposition
        sub_queries = self.decompose_query(rewritten_query)
        
        # 4. Parallel Retrieval
        all_docs, all_metas = self.parallel_retrieve(sub_queries)
        
        if not all_docs:
            logger.warning("[PIPELINE] No documents found!")
            return {"answer": "No relevant documents found. Please ingest documents first.", "sources": []}
        
        # 5. Rerank with rewritten query
        top_docs, top_metas = self.rerank(rewritten_query, all_docs, all_metas, top_k=5)
        
        # 6. Build context
        context = "\n\n---\n\n".join(top_docs)
        sources = list(set(m.get("source", "Unknown") for m in top_metas))
        
        logger.info(f"[PIPELINE] Sources: {sources}")
        
        # Build conversation history for prompt
        conv_context = ""
        if history:
            recent = history[-4:]
            conv_context = "Previous Conversation:\n" + "\n".join([
                f"{msg['role'].upper()}: {msg['content'][:300]}"
                for msg in recent
            ]) + "\n\n"
        
        # 7. Generate with context
        prompt = f"""
        You are a factual question-answering assistant.

        {conv_context}
        Instructions:
        - Answer the question using only the information from the provided documents.
        - Consider the conversation history for context if relevant.

        Guardrails:
        - Use ONLY the information explicitly stated in the provided documents.
        - Do NOT use prior knowledge, assumptions, or external information.

        Requirements:
        - Be clear and concise.
        - Every factual claim must be supported by the provided information.

        Documents:
        {context}

        Question:
        {query}

        Answer:
        """

        try:
            logger.info("[GENERATE] Calling Gemini LLM...")
            answer = self._call_llm(
                prompt,
                temperature=0.3,
                system_instruction="You are a helpful assistant. Answer based on the provided context. If unsure, say so."
            )
            
            result = {"answer": answer, "sources": sources}
            
            # 8. Cache the result
            query_cache.set(query, result, len(history))
            logger.info(f"[CACHE] Response cached. Cache size: {query_cache.stats()['size']}")
            logger.info("=" * 60)
            
            return result
        except Exception as e:
            logger.error(f"[GENERATE] LLM Error: {e}")
            return {"answer": f"Error: {str(e)}", "sources": []}


rag_service = RAGService()
