"""
In-Memory Schema Analysis Cache Service
Handles caching and retrieval of schema analysis results with semantic similarity
"""

import time
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase

# Import performance monitor for integration
try:
    from .performance_monitor_enhanced import performance_monitor
    PERFORMANCE_INTEGRATION = True
except ImportError:
    PERFORMANCE_INTEGRATION = False
    print("⚠️ Performance monitor not available for schema cache")


@dataclass
class CachedAnalysis:
    """Enhanced data class for cached schema analysis with LRU tracking"""
    analysis_data: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: float
    question: str
    context: str
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    question_key: str = ""
    
    def __post_init__(self):
        if not self.question_key:
            self.question_key = hashlib.md5(f"{self.question}|{self.context}".encode()).hexdigest()[:16]
    
    def mark_accessed(self):
        """Mark this entry as recently accessed"""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStatistics:
    """Enhanced cache statistics with performance metrics"""
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    total_queries: int = 0
    evictions: int = 0
    batch_operations: int = 0
    average_similarity: float = 0.0
    total_similarity: float = 0.0
    similarity_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        return (self.exact_hits + self.semantic_hits) / max(self.total_queries, 1) * 100
    
    @property
    def semantic_hit_rate(self) -> float:
        return self.semantic_hits / max(self.total_queries, 1) * 100
    
    def record_similarity(self, similarity: float):
        """Record similarity score for average calculation"""
        self.total_similarity += similarity
        self.similarity_count += 1
        self.average_similarity = self.total_similarity / self.similarity_count


class InMemorySchemaCache:
    """
    Enhanced in-memory cache for schema analysis results with LRU eviction,
    batch processing, and performance monitoring
    """
    
    def __init__(self, 
                 max_size: int = 100, 
                 max_age_hours: int = 24,
                 similarity_threshold: float = 0.85,
                 performance_tracking: bool = True):
        # Enhanced configuration
        self.max_size = max_size
        self.max_age_seconds = max_age_hours * 3600
        self._similarity_threshold = similarity_threshold
        self._performance_tracking = performance_tracking
        
        # Enhanced LRU storage using OrderedDict
        self.exact_cache: OrderedDict[str, CachedAnalysis] = OrderedDict()
        self.semantic_cache: OrderedDict[str, CachedAnalysis] = OrderedDict()
        
        # Enhanced statistics with CacheStatistics class
        self.stats = CacheStatistics()
        
        # Performance monitoring
        self._access_patterns: Dict[str, List[float]] = {}
        self._pending_batch_operations: List[Tuple[str, CachedAnalysis]] = []
        self._batch_size = 10
        self._last_cleanup = time.time()
        self._cleanup_interval = 30 * 60  # 30 minutes
        
        print(f"🗄️ Enhanced Schema Cache initialized: max_size={max_size}, "
              f"similarity_threshold={similarity_threshold}, performance_tracking={performance_tracking}")

    def _evict_lru_entries(self, cache_dict: OrderedDict, target_size: int):
        """Enhanced LRU eviction with performance tracking"""
        evicted_count = 0
        while len(cache_dict) >= target_size:
            # Remove least recently used item (first item in OrderedDict)
            key, entry = cache_dict.popitem(last=False)
            evicted_count += 1
            
            # Clean up access patterns
            if key in self._access_patterns:
                del self._access_patterns[key]
        
        if evicted_count > 0:
            self.stats.evictions += evicted_count
            if self._performance_tracking:
                print(f"� LRU eviction: removed {evicted_count} entries")
    
    def _update_access_pattern(self, key: str):
        """Track access patterns for predictive caching"""
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        
        access_time = time.time()
        self._access_patterns[key].append(access_time)
        
        # Keep only recent access times (last 10 accesses)
        if len(self._access_patterns[key]) > 10:
            self._access_patterns[key] = self._access_patterns[key][-10:]
    
    def _generate_cache_key(self, question: str, context: str) -> str:
        """Generate cache key for exact matching"""
        content = f"{question.strip().lower()}|{context.strip().lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def get_exact_match(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Enhanced exact cache match with integrated performance tracking"""
        
        # Track cache lookup performance
        if PERFORMANCE_INTEGRATION:
            with performance_monitor.track_operation(
                "schema_cache_exact_lookup",
                question_length=len(question),
                context_length=len(context),
                cache_size=len(self.exact_cache)
            ) as metric:
                result = await self._get_exact_match_internal(question, context)
                
                # Add cache-specific metadata
                metric.metadata.update({
                    "cache_hit": result is not None,
                    "cache_type": "exact"
                })
                
                return result
        else:
            return await self._get_exact_match_internal(question, context)
    
    async def _get_exact_match_internal(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Internal exact match implementation"""
        self.stats.total_queries += 1
        
        cache_key = self._generate_cache_key(question, context)
        
        if cache_key in self.exact_cache:
            cached = self.exact_cache[cache_key]
            
            # Check if not expired
            if time.time() - cached.timestamp < self.max_age_seconds:
                # Mark as accessed and move to end (most recently used)
                cached.mark_accessed()
                self.exact_cache.move_to_end(cache_key)
                
                # Update access patterns
                self._update_access_pattern(cache_key)
                
                self.stats.exact_hits += 1
                if self._performance_tracking:
                    print(f"⚡ Exact cache hit (access #{cached.access_count}) for: {question[:30]}...")
                return cached.analysis_data
            else:
                # Remove expired entry
                del self.exact_cache[cache_key]
                if cache_key in self._access_patterns:
                    del self._access_patterns[cache_key]
        
        return None
    
    async def get_semantic_match(self, question: str, context: str, 
                                embedding_service: EmbeddingGeneratorBase,
                                similarity_threshold: float = None) -> Optional[Dict[str, Any]]:
        """Enhanced semantic cache match with LRU tracking and performance monitoring"""
        if similarity_threshold is None:
            similarity_threshold = self._similarity_threshold
            
        if not embedding_service or not self.semantic_cache:
            return None
        
        try:
            # Generate embedding for current question
            query_text = f"{question} {context}".strip()
            embeddings = await embedding_service.generate_embeddings([query_text])
            
            # Handle embedding result properly
            if not self._is_valid_embedding(embeddings):
                return None
            
            # Convert embedding to list
            query_embedding = self._normalize_embedding(embeddings[0] if hasattr(embeddings, '__getitem__') else embeddings)
            
            # Enhanced search through semantic cache with LRU tracking
            best_similarity = 0
            best_match = None
            best_key = None
            
            # Convert to list for iteration while maintaining order
            cache_items = list(self.semantic_cache.items())
            
            for cache_key, cached in cache_items:
                # Check if not expired
                if time.time() - cached.timestamp >= self.max_age_seconds:
                    # Remove expired entry
                    del self.semantic_cache[cache_key]
                    if cache_key in self._access_patterns:
                        del self._access_patterns[cache_key]
                    continue
                
                if cached.embedding:
                    similarity = self._calculate_cosine_similarity(query_embedding, cached.embedding)
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match = cached
                        best_key = cache_key
            
            if best_match and best_key:
                # Mark as accessed and move to end (most recently used)
                best_match.mark_accessed()
                self.semantic_cache.move_to_end(best_key)
                
                # Update access patterns and statistics
                self._update_access_pattern(best_key)
                self.stats.semantic_hits += 1
                self.stats.record_similarity(best_similarity)
                
                if self._performance_tracking:
                    print(f"🧠 Semantic cache hit (similarity: {best_similarity:.3f}, access #{best_match.access_count}) "
                          f"for: {question[:30]}...")
                
                return {
                    "data": best_match.analysis_data,
                    "similarity": best_similarity,
                    "cached_question": best_match.question,
                    "access_count": best_match.access_count
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Enhanced semantic cache lookup failed: {e}")
            return None
    
    async def store_analysis(self, question: str, context: str, analysis_data: Dict[str, Any],
                           embedding_service: Optional[EmbeddingGeneratorBase] = None,
                           batch_mode: bool = False):
        """Enhanced store analysis with batch processing and LRU management"""
        try:
            # Store in exact cache with LRU management
            cache_key = self._generate_cache_key(question, context)
            
            # Generate embedding if service available
            embedding = None
            if embedding_service:
                try:
                    query_text = f"{question} {context}".strip()
                    embeddings = await embedding_service.generate_embeddings([query_text])
                    
                    # Handle embedding result properly
                    if self._is_valid_embedding(embeddings):
                        embedding = self._normalize_embedding(embeddings[0] if hasattr(embeddings, '__getitem__') else embeddings)
                except Exception as e:
                    print(f"⚠️ Failed to generate embedding for cache: {e}")
            
            # Create enhanced cached analysis
            cached = CachedAnalysis(
                analysis_data=analysis_data,
                embedding=embedding,
                timestamp=time.time(),
                question=question,
                context=context
            )
            
            if batch_mode:
                # Add to batch operations queue
                self._pending_batch_operations.append((cache_key, cached))
                
                # Process batch if size reached
                if len(self._pending_batch_operations) >= self._batch_size:
                    await self._process_batch_operations()
                    
                return
            
            # Store immediately with LRU management
            await self._store_single_entry(cache_key, cached)
            
        except Exception as e:
            print(f"⚠️ Failed to store analysis in enhanced cache: {e}")

    async def _store_single_entry(self, cache_key: str, cached: CachedAnalysis):
        """Store single cache entry with LRU management"""
        # Store in exact cache (LRU managed)
        self.exact_cache[cache_key] = cached
        self.exact_cache.move_to_end(cache_key)  # Mark as most recently used
        
        # Store in semantic cache if embedding available (LRU managed)
        if cached.embedding:
            semantic_key = f"sem_{cache_key}"
            self.semantic_cache[semantic_key] = cached
            self.semantic_cache.move_to_end(semantic_key)
        
        # Apply LRU eviction if needed
        if len(self.exact_cache) > self.max_size:
            self._evict_lru_entries(self.exact_cache, self.max_size)
        
        if len(self.semantic_cache) > self.max_size:
            self._evict_lru_entries(self.semantic_cache, self.max_size)
        
        # Perform cleanup if needed
        await self._cleanup_cache()
        
        if self._performance_tracking:
            print(f"💾 Enhanced cache stored: {cached.question[:30]}... "
                  f"(exact: {len(self.exact_cache)}, semantic: {len(self.semantic_cache)})")

    async def _process_batch_operations(self):
        """Process pending batch operations for better performance"""
        if not self._pending_batch_operations:
            return
        
        batch_size = len(self._pending_batch_operations)
        
        # Process all pending operations
        for cache_key, cached in self._pending_batch_operations:
            await self._store_single_entry(cache_key, cached)
        
        # Clear batch queue
        self._pending_batch_operations.clear()
        self.stats.batch_operations += 1
        
        if self._performance_tracking:
            print(f"📦 Batch processed: {batch_size} cache operations")

    async def flush_batch_operations(self):
        """Manually flush any pending batch operations"""
        if self._pending_batch_operations:
            await self._process_batch_operations()
    
    def _is_valid_embedding(self, embeddings) -> bool:
        """Check if embedding result is valid"""
        if embeddings is None:
            return False
        
        # Handle numpy arrays and lists
        try:
            if hasattr(embeddings, '__len__'):
                return len(embeddings) > 0
            return False
        except Exception:
            return False
    
    def _normalize_embedding(self, embedding) -> List[float]:
        """Normalize embedding to list format"""
        if hasattr(embedding, 'tolist'):
            return embedding.tolist()
        elif hasattr(embedding, '__iter__'):
            return list(embedding)
        else:
            return embedding
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0.0 or magnitude2 == 0.0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    async def _cleanup_cache(self):
        """Enhanced cleanup with performance monitoring"""
        try:
            current_time = time.time()
            
            # Skip cleanup if not enough time has passed
            if current_time - self._last_cleanup < self._cleanup_interval:
                return
            
            initial_exact_size = len(self.exact_cache)
            initial_semantic_size = len(self.semantic_cache)
            
            # Cleanup exact cache by age
            expired_keys = [
                key for key, cached in self.exact_cache.items()
                if current_time - cached.timestamp > self.max_age_seconds
            ]
            
            for key in expired_keys:
                del self.exact_cache[key]
                # Clean up access patterns
                if key in self._access_patterns:
                    del self._access_patterns[key]
            
            # Cleanup semantic cache by age
            expired_semantic = []
            for key, cached in list(self.semantic_cache.items()):
                if current_time - cached.timestamp > self.max_age_seconds:
                    expired_semantic.append(key)
            
            for key in expired_semantic:
                del self.semantic_cache[key]
                if key in self._access_patterns:
                    del self._access_patterns[key]
            
            # Update cleanup timestamp
            self._last_cleanup = current_time
            
            # Log cleanup results if tracking enabled
            if self._performance_tracking and (expired_keys or expired_semantic):
                exact_cleaned = len(expired_keys)
                semantic_cleaned = len(expired_semantic)
                print(f"🧹 Cache cleanup: removed {exact_cleaned} exact, {semantic_cleaned} semantic entries "
                      f"(ages: {initial_exact_size}→{len(self.exact_cache)}, {initial_semantic_size}→{len(self.semantic_cache)})")
        
        except Exception as e:
            print(f"⚠️ Cache cleanup error: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "cache_sizes": {
                "exact_cache": len(self.exact_cache),
                "semantic_cache": len(self.semantic_cache),
                "access_patterns": len(self._access_patterns),
                "pending_batch": len(self._pending_batch_operations)
            },
            "hit_rates": {
                "overall_hit_rate": self.stats.hit_rate,
                "exact_hit_rate": (self.stats.exact_hits / max(self.stats.total_queries, 1)) * 100,
                "semantic_hit_rate": self.stats.semantic_hit_rate
            },
            "statistics": {
                "exact_hits": self.stats.exact_hits,
                "semantic_hits": self.stats.semantic_hits,
                "misses": self.stats.misses,
                "total_queries": self.stats.total_queries,
                "evictions": self.stats.evictions,
                "batch_operations": self.stats.batch_operations,
                "average_similarity": round(self.stats.average_similarity, 4)
            },
            "configuration": {
                "max_size": self.max_size,
                "max_age_hours": self.max_age_seconds / 3600,
                "similarity_threshold": self._similarity_threshold,
                "batch_size": self._batch_size,
                "performance_tracking": self._performance_tracking
            }
        }

    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*60)
        print("📊 ENHANCED SCHEMA CACHE PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"🎯 Hit Rates:")
        print(f"   Overall: {stats['hit_rates']['overall_hit_rate']:.1f}%")
        print(f"   Exact: {stats['hit_rates']['exact_hit_rate']:.1f}%")
        print(f"   Semantic: {stats['hit_rates']['semantic_hit_rate']:.1f}%")
        
        print(f"\n📈 Query Statistics:")
        print(f"   Total Queries: {stats['statistics']['total_queries']}")
        print(f"   Exact Hits: {stats['statistics']['exact_hits']}")
        print(f"   Semantic Hits: {stats['statistics']['semantic_hits']}")
        print(f"   Misses: {stats['statistics']['misses']}")
        print(f"   Avg Similarity: {stats['statistics']['average_similarity']}")
        
        print(f"\n🗄️ Cache Sizes:")
        print(f"   Exact Cache: {stats['cache_sizes']['exact_cache']}")
        print(f"   Semantic Cache: {stats['cache_sizes']['semantic_cache']}")
        print(f"   Access Patterns: {stats['cache_sizes']['access_patterns']}")
        
        print(f"\n⚙️ Performance Metrics:")
        print(f"   LRU Evictions: {stats['statistics']['evictions']}")
        print(f"   Batch Operations: {stats['statistics']['batch_operations']}")
        print(f"   Pending Batch: {stats['cache_sizes']['pending_batch']}")
        
        print("="*60)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics (legacy method for compatibility)"""
        stats = self.get_performance_stats()
        return {
            "exact_hits": stats['statistics']['exact_hits'],
            "semantic_hits": stats['statistics']['semantic_hits'], 
            "misses": stats['statistics']['misses'],
            "total_queries": stats['statistics']['total_queries'],
            "hit_rate": stats['hit_rates']['overall_hit_rate'],
            "cache_size": stats['cache_sizes']['exact_cache'],
            "semantic_cache_size": stats['cache_sizes']['semantic_cache']
        }

    def print_statistics(self):
        """Print cache statistics (legacy method for compatibility)"""
        self.print_performance_summary()

    def clear_cache(self):
        """Enhanced cache clearing with performance tracking reset"""
        initial_exact = len(self.exact_cache)
        initial_semantic = len(self.semantic_cache)
        
        self.exact_cache.clear()
        self.semantic_cache.clear()
        self._access_patterns.clear()
        self._pending_batch_operations.clear()
        
        # Reset statistics
        self.stats = CacheStatistics()
        
        if self._performance_tracking:
            print(f"🗑️ Cache cleared: removed {initial_exact} exact, {initial_semantic} semantic entries")

# Create enhanced default cache instance
default_schema_cache = InMemorySchemaCache(
    max_size=100, 
    max_age_hours=24,
    similarity_threshold=0.85,
    performance_tracking=True
)
