"""
Generic Cache Service - Reusable caching with semantic similarity support
Can be used for any type of data caching in AI projects
"""

import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, TypeVar, Generic, Protocol
from dataclasses import dataclass, field
from collections import OrderedDict
from abc import ABC, abstractmethod


T = TypeVar('T')


class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding services"""
    async def generate_embeddings(self, texts: List[str]) -> Any:
        ...


@dataclass
class CacheEntry(Generic[T]):
    """Generic cache entry with metadata"""
    data: T
    key: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def mark_accessed(self):
        """Mark this entry as recently accessed"""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheConfig:
    """Configuration for cache behavior"""
    max_size: int = 100
    ttl_seconds: int = 86400  # 24 hours
    enable_semantic: bool = True
    similarity_threshold: float = 0.85
    enable_stats: bool = True
    batch_size: int = 10
    cleanup_interval: int = 1800  # 30 minutes


class CacheStats:
    """Generic cache statistics"""
    
    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.operations: Dict[str, int] = {}
        self._reset_time = time.time()
    
    def record_hit(self, hit_type: str = "exact"):
        """Record a cache hit"""
        self.hits += 1
        self.operations[f"hit_{hit_type}"] = self.operations.get(f"hit_{hit_type}", 0) + 1
    
    def record_miss(self):
        """Record a cache miss"""
        self.misses += 1
    
    def record_eviction(self, count: int = 1):
        """Record cache evictions"""
        self.evictions += count
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary"""
        uptime = time.time() - self._reset_time
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 2),
            "evictions": self.evictions,
            "operations": self.operations,
            "uptime_seconds": round(uptime, 2)
        }


class CacheStrategy(ABC):
    """Abstract base class for cache strategies"""
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry, current_time: float, config: CacheConfig) -> bool:
        """Determine if an entry should be evicted"""
        pass
    
    @abstractmethod
    def get_eviction_candidates(self, cache: OrderedDict, target_size: int) -> List[str]:
        """Get list of keys to evict"""
        pass


class LRUStrategy(CacheStrategy):
    """Least Recently Used eviction strategy"""
    
    def should_evict(self, entry: CacheEntry, current_time: float, config: CacheConfig) -> bool:
        """Check if entry is expired"""
        return (current_time - entry.timestamp) > config.ttl_seconds
    
    def get_eviction_candidates(self, cache: OrderedDict, target_size: int) -> List[str]:
        """Get LRU candidates for eviction"""
        candidates = []
        current_size = len(cache)
        
        if current_size <= target_size:
            return candidates
        
        # Get oldest entries (first in OrderedDict)
        for key in list(cache.keys())[:current_size - target_size]:
            candidates.append(key)
        
        return candidates


class GenericCache(Generic[T]):
    """
    Generic cache implementation with pluggable strategies
    Supports both exact and semantic matching
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        strategy: Optional[CacheStrategy] = None,
        key_generator: Optional[callable] = None
    ):
        self.config = config or CacheConfig()
        self.strategy = strategy or LRUStrategy()
        self.key_generator = key_generator or self._default_key_generator
        
        # Storage
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._embeddings_index: Dict[str, List[float]] = {}
        
        # Statistics
        self.stats = CacheStats() if self.config.enable_stats else None
        
        # Batch processing
        self._pending_batch: List[Tuple[str, CacheEntry[T]]] = []
        
        # Maintenance
        self._last_cleanup = time.time()
    
    def _default_key_generator(self, *args, **kwargs) -> str:
        """Default key generation from arguments"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def get(
        self, 
        key: str,
        semantic_key: Optional[str] = None,
        embedding_service: Optional[EmbeddingServiceProtocol] = None
    ) -> Optional[T]:
        """Get item from cache with optional semantic search"""
        
        # Try exact match first
        if key in self._cache:
            entry = self._cache[key]
            
            # Check if not expired
            if not self.strategy.should_evict(entry, time.time(), self.config):
                entry.mark_accessed()
                self._cache.move_to_end(key)
                
                if self.stats:
                    self.stats.record_hit("exact")
                
                return entry.data
            else:
                # Remove expired entry
                del self._cache[key]
        
        # Try semantic match if enabled
        if self.config.enable_semantic and semantic_key and embedding_service:
            result = await self._semantic_search(semantic_key, embedding_service)
            if result:
                if self.stats:
                    self.stats.record_hit("semantic")
                return result
        
        if self.stats:
            self.stats.record_miss()
        
        return None
    
    async def _semantic_search(
        self, 
        query: str,
        embedding_service: EmbeddingServiceProtocol
    ) -> Optional[T]:
        """Perform semantic search in cache"""
        try:
            # Generate query embedding
            embeddings = await embedding_service.generate_embeddings([query])
            if not embeddings:
                return None
            
            query_embedding = self._normalize_embedding(embeddings[0])
            
            # Find best match
            best_score = 0.0
            best_entry = None
            
            for key, entry in self._cache.items():
                if entry.embedding:
                    score = self._cosine_similarity(query_embedding, entry.embedding)
                    if score > best_score and score >= self.config.similarity_threshold:
                        best_score = score
                        best_entry = entry
            
            if best_entry:
                best_entry.mark_accessed()
                return best_entry.data
            
        except Exception:
            pass
        
        return None
    
    async def put(
        self,
        key: str,
        value: T,
        metadata: Optional[Dict[str, Any]] = None,
        semantic_key: Optional[str] = None,
        embedding_service: Optional[EmbeddingServiceProtocol] = None,
        batch: bool = False
    ):
        """Store item in cache"""
        
        # Create cache entry
        entry = CacheEntry(
            data=value,
            key=key,
            metadata=metadata or {}
        )
        
        # Generate embedding if semantic search is enabled
        if self.config.enable_semantic and semantic_key and embedding_service:
            try:
                embeddings = await embedding_service.generate_embeddings([semantic_key])
                if embeddings:
                    entry.embedding = self._normalize_embedding(embeddings[0])
            except Exception:
                pass
        
        if batch:
            self._pending_batch.append((key, entry))
            if len(self._pending_batch) >= self.config.batch_size:
                await self.flush_batch()
        else:
            await self._store_entry(key, entry)
    
    async def _store_entry(self, key: str, entry: CacheEntry[T]):
        """Store single entry with eviction if needed"""
        
        # Add to cache
        self._cache[key] = entry
        self._cache.move_to_end(key)
        
        # Handle eviction
        if len(self._cache) > self.config.max_size:
            candidates = self.strategy.get_eviction_candidates(
                self._cache, 
                self.config.max_size
            )
            
            for candidate_key in candidates:
                del self._cache[candidate_key]
            
            if self.stats and candidates:
                self.stats.record_eviction(len(candidates))
        
        # Periodic cleanup
        await self._maybe_cleanup()
    
    async def flush_batch(self):
        """Process pending batch operations"""
        if not self._pending_batch:
            return
        
        for key, entry in self._pending_batch:
            await self._store_entry(key, entry)
        
        self._pending_batch.clear()
    
    async def _maybe_cleanup(self):
        """Perform cleanup if interval has passed"""
        current_time = time.time()
        
        if current_time - self._last_cleanup < self.config.cleanup_interval:
            return
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self._cache.items()
            if self.strategy.should_evict(entry, current_time, self.config)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        self._last_cleanup = current_time
    
    def _normalize_embedding(self, embedding: Any) -> List[float]:
        """Normalize embedding to list format"""
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        else:
            return list(embedding)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._embeddings_index.clear()
        self._pending_batch.clear()
        
        if self.stats:
            self.stats = CacheStats()
    
    def get_info(self) -> Dict[str, Any]:
        """Get cache information"""
        info = {
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "ttl_seconds": self.config.ttl_seconds,
            "semantic_enabled": self.config.enable_semantic
        }
        
        if self.stats:
            info["stats"] = self.stats.get_summary()
        
        return info


# Specialized cache for schema analysis
class SchemaAnalysisCache(GenericCache[Dict[str, Any]]):
    """Specialized cache for schema analysis results"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        super().__init__(
            config=config,
            key_generator=self._schema_key_generator
        )
    
    def _schema_key_generator(self, question: str, context: str = "") -> str:
        """Generate cache key for schema analysis"""
        content = f"{question.strip().lower()}|{context.strip().lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def get_analysis(
        self,
        question: str,
        context: str = "",
        embedding_service: Optional[EmbeddingServiceProtocol] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        key = self._schema_key_generator(question, context)
        semantic_key = f"{question} {context}".strip() if embedding_service else None
        
        return await self.get(key, semantic_key, embedding_service)
    
    async def store_analysis(
        self,
        question: str,
        context: str,
        analysis_data: Dict[str, Any],
        embedding_service: Optional[EmbeddingServiceProtocol] = None,
        batch: bool = False
    ):
        """Store analysis result"""
        key = self._schema_key_generator(question, context)
        semantic_key = f"{question} {context}".strip() if embedding_service else None
        
        metadata = {
            "question": question,
            "context": context,
            "timestamp": time.time()
        }
        
        await self.put(
            key=key,
            value=analysis_data,
            metadata=metadata,
            semantic_key=semantic_key,
            embedding_service=embedding_service,
            batch=batch
        )


# Factory function for easy cache creation
def create_cache(
    cache_type: str = "generic",
    config: Optional[Dict[str, Any]] = None
) -> GenericCache:
    """Factory function to create different cache types"""
    
    cache_config = CacheConfig(**config) if config else CacheConfig()
    
    if cache_type == "schema":
        return SchemaAnalysisCache(cache_config)
    else:
        return GenericCache(cache_config)