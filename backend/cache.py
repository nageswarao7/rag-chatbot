"""
Caching utilities for RAG application.
Implements in-memory LRU cache for query results.
"""
import hashlib
import time
from collections import OrderedDict
from threading import Lock

class LRUCache:
    """
    Thread-safe LRU Cache for caching query results.
    
    Why LRU Cache?
    - Reduces repeated LLM calls for same/similar queries
    - Improves response time for frequent queries
    - Memory-efficient with automatic eviction
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.lock = Lock()
    
    def _hash_key(self, query: str, history_len: int = 0) -> str:
        """Generate cache key from query and history length."""
        key_str = f"{query.strip().lower()}:{history_len}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, history_len: int = 0) -> dict | None:
        """
        Get cached result for query.
        Returns None if not found or expired.
        """
        key = self._hash_key(query, history_len)
        
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl:
                del self.cache[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return entry["value"]
    
    def set(self, query: str, result: dict, history_len: int = 0):
        """Cache a query result."""
        key = self._hash_key(query, history_len)
        
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = {
                "value": result,
                "timestamp": time.time()
            }
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl
            }


# Singleton cache instance
query_cache = LRUCache(max_size=100, ttl_seconds=3600)
