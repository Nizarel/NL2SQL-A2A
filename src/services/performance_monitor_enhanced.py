"""
Performance Monitoring Service
Adds timing and metrics to existing NL2SQL system without breaking functionality
"""
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class PerformanceMetrics:
    """Track performance metrics for NL2SQL operations"""
    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark operation as finished"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API"""
        return {
            'operation': self.operation_name,
            'duration': self.duration,
            'success': self.success,
            'error': self.error_message,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat(),
            'metadata': self.metadata
        }

class PerformanceMonitor:
    """Non-invasive performance monitoring for existing NL2SQL system"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_operations = {}
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def track_operation(self, operation_name: str, **metadata):
        """Context manager to track operation performance"""
        metric = PerformanceMetrics(operation_name, metadata=metadata)
        operation_id = f"{operation_name}_{id(metric)}"
        self.current_operations[operation_id] = metric
        
        try:
            self.logger.info(f"Starting operation: {operation_name}")
            yield metric
            metric.finish(success=True)
            self.logger.info(f"Completed operation: {operation_name} in {metric.duration:.2f}s")
        except Exception as e:
            metric.finish(success=False, error_message=str(e))
            self.logger.error(f"Failed operation: {operation_name} after {metric.duration:.2f}s - {str(e)}")
            raise
        finally:
            self.metrics_history.append(metric)
            self.current_operations.pop(operation_id, None)
    
    def get_recent_metrics(self, limit: int = 10) -> list[Dict[str, Any]]:
        """Get recent performance metrics"""
        return [m.to_dict() for m in self.metrics_history[-limit:]]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation type"""
        ops = [m for m in self.metrics_history if m.operation_name == operation_name and m.duration is not None]
        if not ops:
            return {"operation": operation_name, "count": 0}
        
        durations = [op.duration for op in ops]
        success_count = sum(1 for op in ops if op.success)
        
        return {
            "operation": operation_name,
            "count": len(ops),
            "success_rate": success_count / len(ops),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "recent_duration": durations[-1] if durations else None
        }
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system performance overview"""
        if not self.metrics_history:
            return {"status": "no_data", "total_operations": 0}
        
        # Get unique operation types
        operation_types = set(m.operation_name for m in self.metrics_history)
        
        # Calculate overall stats
        total_ops = len(self.metrics_history)
        successful_ops = sum(1 for m in self.metrics_history if m.success)
        recent_ops = self.metrics_history[-5:] if len(self.metrics_history) >= 5 else self.metrics_history
        
        return {
            "status": "active",
            "total_operations": total_ops,
            "success_rate": successful_ops / total_ops,
            "operation_types": list(operation_types),
            "recent_operations": [m.to_dict() for m in recent_ops],
            "stats_by_operation": {op: self.get_operation_stats(op) for op in operation_types}
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
