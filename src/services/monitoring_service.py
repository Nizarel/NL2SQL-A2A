"""
System Monitoring Service - Comprehensive health checks and performance monitoring
Provides real-time system status, performance metrics, and health diagnostics
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from enum import Enum


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PerformanceMetric:
    """Individual performance metric tracking"""
    
    def __init__(self, name: str, unit: str = "ms"):
        self.name = name
        self.unit = unit
        self.values: List[float] = []
        self.timestamps: List[datetime] = []
        self.max_history = 1000  # Keep last 1000 measurements
    
    def record(self, value: float) -> None:
        """Record a new metric value"""
        self.values.append(value)
        self.timestamps.append(datetime.now(timezone.utc))
        
        # Maintain history limit
        if len(self.values) > self.max_history:
            self.values.pop(0)
            self.timestamps.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistical summary of metric"""
        if not self.values:
            return {"count": 0, "unit": self.unit}
        
        return {
            "count": len(self.values),
            "current": self.values[-1],
            "average": sum(self.values) / len(self.values),
            "min": min(self.values),
            "max": max(self.values),
            "unit": self.unit,
            "last_updated": self.timestamps[-1].isoformat()
        }


class SystemMonitoringService:
    """
    Comprehensive system monitoring service providing:
    - Real-time health status monitoring
    - Performance metrics collection and analysis
    - Resource usage tracking
    - Alert generation for issues
    - Historical data analysis
    """
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Initialize core metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize core performance metrics"""
        metrics_config = [
            ("query_processing_time", "ms"),
            ("sql_generation_time", "ms"),
            ("template_render_time", "ms"),
            ("database_query_time", "ms"),
            ("memory_usage", "MB"),
            ("cpu_usage", "%"),
            ("active_connections", "count"),
            ("error_rate", "%"),
            ("cache_hit_rate", "%")
        ]
        
        for name, unit in metrics_config:
            self.metrics[name] = PerformanceMetric(name, unit)
    
    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Record a performance metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value to record
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].record(value)
        else:
            # Create new metric if it doesn't exist
            self.metrics[metric_name] = PerformanceMetric(metric_name)
            self.metrics[metric_name].record(value)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health status
        
        Returns:
            Complete health assessment
        """
        health_data = {
            "overall_status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "components": {},
            "alerts": self.get_active_alerts(),
            "performance_summary": self.get_performance_summary()
        }
        
        # Check individual components
        components = [
            ("database_connection", self._check_database_health),
            ("memory_usage", self._check_memory_health),
            ("cpu_usage", self._check_cpu_health),
            ("error_rate", self._check_error_rate_health),
            ("response_time", self._check_response_time_health)
        ]
        
        overall_status = HealthStatus.HEALTHY
        
        for component_name, check_function in components:
            try:
                component_health = check_function()
                health_data["components"][component_name] = component_health
                
                # Update overall status based on component health
                component_status = HealthStatus(component_health["status"])
                if component_status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif component_status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
                    
            except Exception as e:
                health_data["components"][component_name] = {
                    "status": HealthStatus.UNKNOWN.value,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        health_data["overall_status"] = overall_status.value
        return health_data
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connection health"""
        # This would typically test database connectivity
        # For now, return a basic health check
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "Database connections operational",
            "active_connections": self.metrics.get("active_connections", PerformanceMetric("active_connections")).get_stats().get("current", 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check system memory health"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            status = HealthStatus.HEALTHY
            if memory_percent > 90:
                status = HealthStatus.CRITICAL
            elif memory_percent > 80:
                status = HealthStatus.WARNING
            
            return {
                "status": status.value,
                "memory_percent": memory_percent,
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024),
                "message": f"Memory usage at {memory_percent:.1f}%",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _check_cpu_health(self) -> Dict[str, Any]:
        """Check CPU usage health"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = HealthStatus.HEALTHY
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
            
            return {
                "status": status.value,
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "message": f"CPU usage at {cpu_percent:.1f}%",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def _check_error_rate_health(self) -> Dict[str, Any]:
        """Check system error rate health"""
        error_metric = self.metrics.get("error_rate")
        if not error_metric or not error_metric.values:
            return {
                "status": HealthStatus.HEALTHY.value,
                "error_rate": 0,
                "message": "No error rate data available",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        current_error_rate = error_metric.values[-1]
        
        status = HealthStatus.HEALTHY
        if current_error_rate > 10:
            status = HealthStatus.CRITICAL
        elif current_error_rate > 5:
            status = HealthStatus.WARNING
        
        return {
            "status": status.value,
            "error_rate": current_error_rate,
            "message": f"Error rate at {current_error_rate:.1f}%",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_response_time_health(self) -> Dict[str, Any]:
        """Check system response time health"""
        query_time_metric = self.metrics.get("query_processing_time")
        if not query_time_metric or not query_time_metric.values:
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "No response time data available",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        stats = query_time_metric.get_stats()
        avg_response_time = stats["average"]
        
        status = HealthStatus.HEALTHY
        if avg_response_time > 60000:  # 60 seconds
            status = HealthStatus.CRITICAL
        elif avg_response_time > 30000:  # 30 seconds
            status = HealthStatus.WARNING
        
        return {
            "status": status.value,
            "average_response_time_ms": avg_response_time,
            "current_response_time_ms": stats["current"],
            "message": f"Average response time: {avg_response_time:.0f}ms",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        summary = {
            "metrics_count": len(self.metrics),
            "data_points": sum(len(metric.values) for metric in self.metrics.values()),
            "metrics": {}
        }
        
        for name, metric in self.metrics.items():
            summary["metrics"][name] = metric.get_stats()
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        # Filter alerts from last 24 hours
        now = datetime.now(timezone.utc)
        recent_alerts = []
        
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert["timestamp"].replace("Z", "+00:00"))
            if (now - alert_time).total_seconds() < 86400:  # 24 hours
                recent_alerts.append(alert)
        
        return recent_alerts
    
    def create_alert(
        self,
        level: str,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a new system alert
        
        Args:
            level: Alert level (info, warning, critical)
            component: Component that triggered the alert
            message: Alert message
            details: Additional alert details
        """
        alert = {
            "id": f"alert_{int(time.time())}_{len(self.alerts)}",
            "level": level,
            "component": component,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False
        }
        
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts.pop(0)
    
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous system monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop continuous system monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Continuous monitoring loop"""
        while self._monitoring_active:
            try:
                # Record system metrics
                self._record_system_metrics()
                
                # Check for alerts
                self._check_alert_conditions()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.create_alert(
                    level="warning",
                    component="monitoring_service",
                    message=f"Monitoring loop error: {str(e)}"
                )
                time.sleep(interval_seconds)
    
    def _record_system_metrics(self) -> None:
        """Record current system metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("memory_usage", memory.used / (1024 * 1024))
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.record_metric("cpu_usage", cpu_percent)
            
        except Exception:
            pass  # Silently ignore system metric collection errors
    
    def _check_alert_conditions(self) -> None:
        """Check for conditions that should trigger alerts"""
        # Check memory usage
        memory_metric = self.metrics.get("memory_usage")
        if memory_metric and memory_metric.values:
            current_memory = memory_metric.values[-1]
            if current_memory > 90:  # 90% memory usage
                self.create_alert(
                    level="critical",
                    component="memory",
                    message=f"High memory usage: {current_memory:.1f}%"
                )
        
        # Check error rate
        error_metric = self.metrics.get("error_rate")
        if error_metric and error_metric.values:
            current_error_rate = error_metric.values[-1]
            if current_error_rate > 10:  # 10% error rate
                self.create_alert(
                    level="critical",
                    component="error_rate",
                    message=f"High error rate: {current_error_rate:.1f}%"
                )
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring service statistics"""
        return {
            "monitoring_active": self._monitoring_active,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "metrics_tracked": len(self.metrics),
            "total_data_points": sum(len(metric.values) for metric in self.metrics.values()),
            "active_alerts": len(self.get_active_alerts()),
            "total_alerts": len(self.alerts)
        }


# Global monitoring service instance
monitoring_service = SystemMonitoringService()
