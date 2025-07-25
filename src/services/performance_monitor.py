"""
Performance monitoring utilities for the NL2SQL system
"""

import time
import functools
from typing import Callable, Any, Dict, List
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque


class PerformanceMetric:
    """Individual performance metric tracker"""
    
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.total_time = 0.0
        self.success_count = 0
        self.error_count = 0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.recent_times = deque(maxlen=100)  # Keep last 100 execution times
    
    def record(self, elapsed_time: float, success: bool = True):
        """Record a new measurement"""
        self.count += 1
        self.total_time += elapsed_time
        self.recent_times.append(elapsed_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
    
    @property
    def avg_time(self) -> float:
        """Average execution time"""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Success rate percentage"""
        return (self.success_count / self.count * 100) if self.count > 0 else 0.0
    
    @property
    def recent_avg_time(self) -> float:
        """Average of recent execution times"""
        return sum(self.recent_times) / len(self.recent_times) if self.recent_times else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "count": self.count,
            "total_time": round(self.total_time, 3),
            "avg_time": round(self.avg_time, 3),
            "recent_avg_time": round(self.recent_avg_time, 3),
            "min_time": round(self.min_time, 3) if self.min_time != float('inf') else 0,
            "max_time": round(self.max_time, 3),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 1)
        }


class PerformanceMonitor:
    """Performance monitoring for async operations"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = {}
        self._enabled = True
        self._start_time = datetime.now()
    
    def enable(self):
        """Enable performance monitoring"""
        self._enabled = True
    
    def disable(self):
        """Disable performance monitoring"""
        self._enabled = False
    
    def track_async(self, name: str, include_args: bool = False):
        """
        Decorator for tracking async function performance
        
        Args:
            name: Name of the metric to track
            include_args: Whether to include function arguments in error logs
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                if not self._enabled:
                    return await func(*args, **kwargs)
                
                start = time.time()
                success = True
                error_details = None
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Check if result indicates failure
                    if isinstance(result, dict) and not result.get("success", True):
                        success = False
                        error_details = result.get("error", "Unknown error")
                    
                    return result
                    
                except Exception as e:
                    success = False
                    error_details = str(e)
                    raise
                    
                finally:
                    elapsed = time.time() - start
                    self._record_metric(name, elapsed, success, error_details)
            
            return wrapper
        return decorator
    
    def track_sync(self, name: str):
        """Decorator for tracking synchronous function performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if not self._enabled:
                    return func(*args, **kwargs)
                
                start = time.time()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception:
                    success = False
                    raise
                finally:
                    elapsed = time.time() - start
                    self._record_metric(name, elapsed, success)
            
            return wrapper
        return decorator
    
    def record_manual(self, name: str, elapsed_time: float, success: bool = True):
        """Manually record a performance metric"""
        if self._enabled:
            self._record_metric(name, elapsed_time, success)
    
    def _record_metric(self, name: str, elapsed: float, success: bool, error_details: str = None):
        """Record performance metric"""
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetric(name)
        
        self.metrics[name].record(elapsed, success)
        
        # Log slow operations (>5 seconds)
        if elapsed > 5.0:
            print(f"⚠️ Slow operation detected: {name} took {elapsed:.2f}s")
            if not success and error_details:
                print(f"   Error: {error_details}")
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionary"""
        return {name: metric.to_dict() for name, metric in self.metrics.items()}
    
    def get_metric(self, name: str) -> Dict[str, Any]:
        """Get specific metric"""
        if name in self.metrics:
            return self.metrics[name].to_dict()
        return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {"message": "No metrics recorded"}
        
        total_operations = sum(metric.count for metric in self.metrics.values())
        total_success = sum(metric.success_count for metric in self.metrics.values())
        total_time = sum(metric.total_time for metric in self.metrics.values())
        
        # Find slowest operations
        slowest_ops = sorted(
            [(name, metric.avg_time) for name, metric in self.metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Find operations with most errors
        error_prone_ops = sorted(
            [(name, metric.error_count) for name, metric in self.metrics.items() if metric.error_count > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        uptime = datetime.now() - self._start_time
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "total_operations": total_operations,
            "total_success_rate": round((total_success / total_operations * 100) if total_operations > 0 else 0, 1),
            "total_processing_time": round(total_time, 3),
            "avg_operation_time": round(total_time / total_operations, 3) if total_operations > 0 else 0,
            "slowest_operations": [{"name": name, "avg_time": round(time, 3)} for name, time in slowest_ops],
            "error_prone_operations": [{"name": name, "error_count": count} for name, count in error_prone_ops],
            "metrics_count": len(self.metrics)
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self._start_time = datetime.now()
    
    def print_summary(self):
        """Print performance summary to console"""
        summary = self.get_summary()
        
        print("\n📊 Performance Summary")
        print("=" * 50)
        print(f"Uptime: {summary['uptime_seconds']:.1f}s")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Success Rate: {summary['total_success_rate']}%")
        print(f"Avg Operation Time: {summary['avg_operation_time']}s")
        
        if summary['slowest_operations']:
            print("\n🐌 Slowest Operations:")
            for op in summary['slowest_operations']:
                print(f"  • {op['name']}: {op['avg_time']}s")
        
        if summary['error_prone_operations']:
            print("\n❌ Error-Prone Operations:")
            for op in summary['error_prone_operations']:
                print(f"  • {op['name']}: {op['error_count']} errors")


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


# Convenience decorators using global monitor
def track_async_performance(name: str, include_args: bool = False):
    """Convenience decorator using global performance monitor"""
    return perf_monitor.track_async(name, include_args)


def track_sync_performance(name: str):
    """Convenience decorator for sync functions using global performance monitor"""
    return perf_monitor.track_sync(name)
