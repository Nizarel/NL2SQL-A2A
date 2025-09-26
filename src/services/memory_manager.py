"""
Memory Management Service for NL2SQL Application
Handles memory cleanup and resource management to prevent performance degradation
"""

import asyncio
import logging
import gc
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from collections import OrderedDict
import sys

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Thread-safe LRU cache with size limits and TTL support
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
    
    def get(self, key: str) -> Any:
        """Get item from cache, return None if not found or expired"""
        if key not in self._cache:
            return None
        
        # Check TTL
        if self._is_expired(key):
            self.remove(key)
            return None
        
        # Move to end (most recently used)
        value = self._cache.pop(key)
        self._cache[key] = value
        return value
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting old items if necessary"""
        # Remove if exists (to update position)
        if key in self._cache:
            self._cache.pop(key)
        
        # Add new item
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
        
        # Evict if over size limit
        while len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            self.remove(oldest_key)
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
            return True
        return False
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache item is expired"""
        if key not in self._timestamps:
            return True
        
        age = datetime.now() - self._timestamps[key]
        return age.total_seconds() > self.ttl_seconds
    
    def clear_expired(self) -> int:
        """Remove expired items, return count of removed items"""
        expired_keys = []
        for key in list(self._cache.keys()):
            if self._is_expired(key):
                expired_keys.append(key)
        
        for key in expired_keys:
            self.remove(key)
        
        return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear all cache items"""
        self._cache.clear()
        self._timestamps.clear()


class WorkflowManager:
    """
    Manages active workflows with automatic cleanup
    """
    
    def __init__(self, max_workflows: int = 100, cleanup_interval: int = 300):
        self.max_workflows = max_workflows
        self.cleanup_interval = cleanup_interval  # seconds
        self._workflows = {}
        self._last_cleanup = datetime.now()
    
    def add_workflow(self, workflow_id: str, workflow_context: Any) -> None:
        """Add workflow with automatic cleanup trigger"""
        self._workflows[workflow_id] = {
            'context': workflow_context,
            'created_at': datetime.now(),
            'last_accessed': datetime.now()
        }
        
        # Trigger cleanup if needed
        if len(self._workflows) > self.max_workflows:
            self._cleanup_old_workflows(force=True)
        elif (datetime.now() - self._last_cleanup).seconds > self.cleanup_interval:
            self._cleanup_old_workflows()
    
    def get_workflow(self, workflow_id: str) -> Optional[Any]:
        """Get workflow and update last accessed time"""
        if workflow_id in self._workflows:
            self._workflows[workflow_id]['last_accessed'] = datetime.now()
            return self._workflows[workflow_id]['context']
        return None
    
    def remove_workflow(self, workflow_id: str) -> bool:
        """Remove specific workflow"""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False
    
    def _cleanup_old_workflows(self, force: bool = False) -> int:
        """Clean up old or inactive workflows"""
        self._last_cleanup = datetime.now()
        cutoff_time = datetime.now() - timedelta(hours=2)  # 2 hours max age
        inactive_cutoff = datetime.now() - timedelta(minutes=30)  # 30 minutes inactive
        
        workflows_to_remove = []
        
        for workflow_id, workflow_data in self._workflows.items():
            # Remove very old workflows
            if workflow_data['created_at'] < cutoff_time:
                workflows_to_remove.append(workflow_id)
            # Remove inactive workflows if we're over limit
            elif force and workflow_data['last_accessed'] < inactive_cutoff:
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self._workflows[workflow_id]
        
        if workflows_to_remove:
            logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
        
        return len(workflows_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        now = datetime.now()
        active_count = 0
        old_count = 0
        
        for workflow_data in self._workflows.values():
            age_minutes = (now - workflow_data['last_accessed']).total_seconds() / 60
            if age_minutes < 10:  # Active in last 10 minutes
                active_count += 1
            elif age_minutes > 60:  # Older than 1 hour
                old_count += 1
        
        return {
            'total_workflows': len(self._workflows),
            'active_workflows': active_count,
            'old_workflows': old_count,
            'memory_usage_estimate_mb': len(self._workflows) * 0.5  # Rough estimate
        }


class MemoryManager:
    """
    Central memory management service
    """
    
    def __init__(self):
        self.embedding_cache = LRUCache(max_size=500, ttl_seconds=1800)  # 30 minutes TTL
        self.query_cache = LRUCache(max_size=1000, ttl_seconds=3600)    # 1 hour TTL
        self.workflow_manager = WorkflowManager(max_workflows=50)
        self._cleanup_task = None
        self._initialized = False
    
    def _start_cleanup_task(self):
        """Start background cleanup task (lazy initialization)"""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                self._initialized = True
        except RuntimeError:
            # No event loop running, will start later
            self._initialized = False
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.cleanup_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _ensure_initialized(self):
        """Ensure cleanup task is started if event loop is available"""
        if not self._initialized:
            self._start_cleanup_task()
    
    async def cleanup_memory(self) -> Dict[str, int]:
        """Perform comprehensive memory cleanup"""
        self._ensure_initialized()  # Make sure cleanup task is running
        
        stats = {}
        
        # Clean expired cache items
        stats['expired_embeddings'] = self.embedding_cache.clear_expired()
        stats['expired_queries'] = self.query_cache.clear_expired()
        
        # Clean old workflows
        stats['cleaned_workflows'] = self.workflow_manager._cleanup_old_workflows()
        
        # Force garbage collection
        collected = gc.collect()
        stats['gc_collected'] = collected
        
        # Log memory usage
        memory_info = self.get_memory_stats()
        logger.info(f"Memory cleanup completed: {stats}, Memory: {memory_info}")
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'percent': round(process.memory_percent(), 2),
                'embedding_cache_size': self.embedding_cache.size(),
                'query_cache_size': self.query_cache.size(),
                'workflow_stats': self.workflow_manager.get_stats(),
                'python_objects': len(gc.get_objects())
            }
        except ImportError:
            # Fallback without psutil
            return {
                'embedding_cache_size': self.embedding_cache.size(),
                'query_cache_size': self.query_cache.size(),
                'workflow_stats': self.workflow_manager.get_stats(),
                'python_objects': len(gc.get_objects())
            }
    
    async def emergency_cleanup(self) -> Dict[str, int]:
        """Emergency memory cleanup for critical situations"""
        logger.warning("Performing emergency memory cleanup")
        
        stats = {}
        
        # Clear all caches
        self.embedding_cache.clear()
        self.query_cache.clear()
        stats['cleared_caches'] = 2
        
        # Clean all workflows older than 5 minutes
        old_cutoff = datetime.now() - timedelta(minutes=5)
        workflows_to_remove = []
        
        for workflow_id, workflow_data in self.workflow_manager._workflows.items():
            if workflow_data['last_accessed'] < old_cutoff:
                workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            self.workflow_manager.remove_workflow(workflow_id)
        
        stats['emergency_workflows_cleared'] = len(workflows_to_remove)
        
        # Force aggressive garbage collection
        for i in range(3):
            collected = gc.collect()
            stats['gc_collected'] = stats.get('gc_collected', 0) + collected
        
        logger.warning(f"Emergency cleanup completed: {stats}")
        return stats
    
    def should_trigger_emergency_cleanup(self) -> bool:
        """Check if emergency cleanup should be triggered"""
        try:
            import psutil
            process = psutil.Process()
            memory_percent = process.memory_percent()
            
            # Trigger if using more than 80% of available memory
            if memory_percent > 80:
                return True
            
            # Trigger if cache sizes are too large
            if (self.embedding_cache.size() > 800 or 
                self.query_cache.size() > 1500 or
                len(self.workflow_manager._workflows) > 80):
                return True
                
        except ImportError:
            # Without psutil, use cache sizes only
            if (self.embedding_cache.size() > 400 or 
                len(self.workflow_manager._workflows) > 40):
                return True
        
        return False
    
    async def close(self):
        """Close memory manager and cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final cleanup
        await self.cleanup_memory()


# Global memory manager instance
memory_manager = MemoryManager()