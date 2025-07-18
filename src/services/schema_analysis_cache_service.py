"""
In-Memory Schema Analysis Cache Service
Handles caching and retrieval of schema analysis results with semantic similarity
"""

import time
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase


@dataclass
class CachedAnalysis:
    """Data class for cached schema analysis"""
    analysis_data: Dict[str, Any]
    embedding: Optional[List[float]]
    timestamp: float
    question: str
    context: str


class InMemorySchemaCache:
    """
    In-memory cache service for schema analysis results with semantic similarity
    """
    
    def __init__(self, max_size: int = 100, max_age_hours: int = 24):
        self.max_size = max_size
        self.max_age_seconds = max_age_hours * 3600
        
        # Storage
        self.exact_cache: Dict[str, CachedAnalysis] = {}
        self.semantic_cache: List[CachedAnalysis] = []
        
        # Statistics
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        
        print("🗄️ In-Memory Schema Cache initialized")
    
    def _generate_cache_key(self, question: str, context: str) -> str:
        """Generate cache key for exact matching"""
        content = f"{question.strip().lower()}|{context.strip().lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def get_exact_match(self, question: str, context: str = "") -> Optional[Dict[str, Any]]:
        """Get exact cache match"""
        self.stats["total_queries"] += 1
        
        cache_key = self._generate_cache_key(question, context)
        
        if cache_key in self.exact_cache:
            cached = self.exact_cache[cache_key]
            
            # Check if not expired
            if time.time() - cached.timestamp < self.max_age_seconds:
                self.stats["exact_hits"] += 1
                print(f"⚡ Exact cache hit for: {question[:30]}...")
                return cached.analysis_data
            else:
                # Remove expired entry
                del self.exact_cache[cache_key]
        
        return None
    
    async def get_semantic_match(self, question: str, context: str, 
                                embedding_service: EmbeddingGeneratorBase,
                                similarity_threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """Get semantically similar cache match"""
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
            
            # Search through semantic cache
            best_similarity = 0
            best_match = None
            
            for cached in self.semantic_cache:
                # Check if not expired
                if time.time() - cached.timestamp >= self.max_age_seconds:
                    continue
                
                if cached.embedding:
                    similarity = self._calculate_cosine_similarity(query_embedding, cached.embedding)
                    
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match = cached
            
            if best_match:
                self.stats["semantic_hits"] += 1
                print(f"🧠 Semantic cache hit (similarity: {best_similarity:.3f}) for: {question[:30]}...")
                return {
                    "data": best_match.analysis_data,
                    "similarity": best_similarity,
                    "cached_question": best_match.question
                }
            
            return None
            
        except Exception as e:
            print(f"⚠️ Semantic cache lookup failed: {e}")
            return None
    
    async def store_analysis(self, question: str, context: str, analysis_data: Dict[str, Any],
                           embedding_service: Optional[EmbeddingGeneratorBase] = None):
        """Store analysis result in cache"""
        try:
            # Store in exact cache
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
            
            # Create cached analysis
            cached = CachedAnalysis(
                analysis_data=analysis_data,
                embedding=embedding,
                timestamp=time.time(),
                question=question,
                context=context
            )
            
            # Store in exact cache
            self.exact_cache[cache_key] = cached
            
            # Store in semantic cache if embedding available
            if embedding:
                self.semantic_cache.append(cached)
            
            # Cleanup if needed
            await self._cleanup_cache()
            
            print(f"💾 Cached analysis for: {question[:30]}...")
            
        except Exception as e:
            print(f"⚠️ Failed to store analysis in cache: {e}")
    
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
        """Cleanup old cache entries"""
        try:
            current_time = time.time()
            
            # Cleanup exact cache by age
            expired_keys = [
                key for key, cached in self.exact_cache.items()
                if current_time - cached.timestamp > self.max_age_seconds
            ]
            
            for key in expired_keys:
                del self.exact_cache[key]
            
            # Cleanup exact cache by size
            if len(self.exact_cache) > self.max_size:
                # Keep most recent entries
                sorted_items = sorted(
                    self.exact_cache.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True
                )
                self.exact_cache = dict(sorted_items[:self.max_size])
            
            # Cleanup semantic cache
            self.semantic_cache = [
                cached for cached in self.semantic_cache
                if current_time - cached.timestamp < self.max_age_seconds
            ]
            
            # Keep only most recent semantic entries
            if len(self.semantic_cache) > self.max_size:
                self.semantic_cache.sort(key=lambda x: x.timestamp, reverse=True)
                self.semantic_cache = self.semantic_cache[:self.max_size]
                
        except Exception as e:
            print(f"⚠️ Cache cleanup failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_queries = self.stats["total_queries"]
        total_hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        hit_rate = (total_hits / max(total_queries, 1)) * 100
        
        return {
            "total_queries": total_queries,
            "exact_hits": self.stats["exact_hits"],
            "semantic_hits": self.stats["semantic_hits"],
            "misses": self.stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "exact_cache_size": len(self.exact_cache),
            "semantic_cache_size": len(self.semantic_cache)
        }
    
    def clear_cache(self):
        """Clear all cache"""
        self.exact_cache.clear()
        self.semantic_cache.clear()
        self.stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "total_queries": 0
        }
        print("🧹 Schema analysis cache cleared")
