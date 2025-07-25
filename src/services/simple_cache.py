"""
Simple Query Result Cache for NL2SQL System
Phase 2 Optimization: Additive caching without breaking existing functionality
"""
import hashlib
import time
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = None
    ttl: float = 3600  # 1 hour default TTL
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

class SimpleQueryCache:
    """
    Simple LRU cache for query results
    Safe, additive optimization that won't break existing functionality
    """
    
    def __init__(self, max_size: int = 100, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def _generate_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate cache key from query and parameters"""
        key_data = {
            "query": query.strip().lower(),
            "params": params or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached result if available and not expired"""
        self.stats["total_requests"] += 1
        
        key = self._generate_key(query, params)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            entry.touch()
            
            self.stats["hits"] += 1
            return entry.data
        
        self.stats["misses"] += 1
        return None
    
    def put(self, query: str, result: Any, params: Dict[str, Any] = None, ttl: float = None) -> None:
        """Store result in cache"""
        key = self._generate_key(query, params)
        ttl = ttl or self.default_ttl
        
        # Remove oldest entries if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
        
        # Store new entry
        entry = CacheEntry(
            data=result,
            timestamp=time.time(),
            ttl=ttl
        )
        entry.touch()
        
        self.cache[key] = entry
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats["total_requests"]
        hit_rate = (self.stats["hits"] / total_requests) if total_requests > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "total_requests": total_requests,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "status": "active" if total_requests > 0 else "inactive"
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        entries_info = []
        current_time = time.time()
        
        for key, entry in list(self.cache.items())[-10:]:  # Last 10 entries
            entries_info.append({
                "key": key[:12] + "...",  # Truncated key
                "age_seconds": current_time - entry.timestamp,
                "access_count": entry.access_count,
                "expired": entry.is_expired(),
                "size_estimate": len(str(entry.data)) if entry.data else 0
            })
        
        stats = self.get_stats()
        return {
            "stats": stats,
            "recent_entries": entries_info,
            "cache_health": "healthy" if stats["hit_rate"] > 0.2 else "warming_up"
        }

# Global cache instances for different types of queries
query_result_cache = SimpleQueryCache(max_size=50, default_ttl=1800)  # 30 minutes
schema_cache = SimpleQueryCache(max_size=20, default_ttl=7200)  # 2 hours
sql_cache = SimpleQueryCache(max_size=100, default_ttl=3600)  # 1 hour
