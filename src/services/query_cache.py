"""
Query Cache Service for NL2SQL Multi-Agent System
Implements LRU caching with TTL for query results
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from config import get_settings


class QueryCache:
    """
    LRU Cache with TTL for query results
    Thread-safe implementation using asyncio locks
    """
    
    def __init__(self, ttl_seconds: Optional[int] = None, max_size: Optional[int] = None):
        settings = get_settings()
        self.ttl = timedelta(seconds=ttl_seconds or settings.query_cache_ttl)
        self.max_size = max_size or settings.query_cache_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.access_order: Dict[str, datetime] = {}  # For LRU tracking
        self._lock = asyncio.Lock()
        self.enabled = settings.enable_query_cache
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _get_cache_key(self, question: str, context: str = "", execute: bool = True, 
                      limit: int = 100) -> str:
        """
        Generate cache key from query parameters
        """
        # Include relevant parameters in cache key
        content = f"{question.lower().strip()}:{context.lower().strip()}:{execute}:{limit}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, question: str, context: str = "", execute: bool = True, 
                  limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Get cached result if available and not expired
        """
        if not self.enabled:
            return None
            
        async with self._lock:
            key = self._get_cache_key(question, context, execute, limit)
            
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check if expired
            if datetime.now() - self.timestamps[key] > self.ttl:
                await self._remove_key(key)
                self.misses += 1
                return None
            
            # Update access time for LRU
            self.access_order[key] = datetime.now()
            self.hits += 1
            
            # Return a copy to prevent external modifications
            return self.cache[key].copy()
    
    async def set(self, question: str, result: Dict[str, Any], context: str = "", 
                  execute: bool = True, limit: int = 100):
        """
        Cache a query result
        """
        if not self.enabled:
            return
            
        async with self._lock:
            key = self._get_cache_key(question, context, execute, limit)
            
            # Don't cache failed results or very large results
            if (not result.get("success", False) or 
                len(str(result)) > get_settings().max_response_size):
                return
            
            # Implement LRU eviction if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            # Store result with timestamp
            self.cache[key] = result.copy()
            self.timestamps[key] = datetime.now()
            self.access_order[key] = datetime.now()
    
    async def _evict_lru(self):
        """
        Evict least recently used item
        """
        if not self.access_order:
            return
            
        # Find least recently used key
        lru_key = min(self.access_order, key=self.access_order.get)
        await self._remove_key(lru_key)
        self.evictions += 1
    
    async def _remove_key(self, key: str):
        """
        Remove a key from all tracking structures
        """
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_order.pop(key, None)
    
    async def clear(self):
        """
        Clear all cached entries
        """
        async with self._lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    async def cleanup_expired(self):
        """
        Remove all expired entries
        """
        if not self.enabled:
            return
            
        async with self._lock:
            current_time = datetime.now()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                await self._remove_key(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "evictions": self.evictions,
            "ttl_seconds": self.ttl.total_seconds()
        }
    
    async def invalidate_pattern(self, pattern: str):
        """
        Invalidate cache entries matching a pattern (simple substring match)
        """
        async with self._lock:
            keys_to_remove = [
                key for key in self.cache.keys()
                if pattern.lower() in key.lower()
            ]
            
            for key in keys_to_remove:
                await self._remove_key(key)
