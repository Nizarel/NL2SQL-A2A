"""
Optimization Dashboard Service
Provides real-time optimization status and performance tracking
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass
from services.performance_monitor_enhanced import performance_monitor
from services.schema_analysis_cache_service import default_schema_cache


@dataclass
class OptimizationPhase:
    """Track optimization phase status"""
    phase_name: str
    description: str
    status: str  # "not_started", "in_progress", "completed", "failed"
    start_time: float = 0.0
    completion_time: float = 0.0
    expected_improvement: str = ""
    actual_improvement: str = ""
    
    @property
    def duration(self) -> float:
        if self.completion_time > 0:
            return self.completion_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


class OptimizationTracker:
    """Track and monitor optimization progress across all phases"""
    
    def __init__(self):
        self.phases = {
            "phase1_monitoring": OptimizationPhase(
                "Phase 1: Performance Monitoring",
                "Add performance tracking to existing system",
                "completed",  # We've implemented this
                expected_improvement="Baseline measurement capability"
            ),
            "phase1_caching": OptimizationPhase(
                "Phase 1: Basic Caching",
                "Implement schema and query result caching",
                "in_progress",  # Currently implementing
                expected_improvement="10-20% performance improvement"
            ),
            "phase2_agent_optimization": OptimizationPhase(
                "Phase 2: Agent-Level Optimizations",
                "Optimize individual agent performance",
                "not_started",
                expected_improvement="20-30% performance improvement"
            ),
            "phase3_parallel_execution": OptimizationPhase(
                "Phase 3: Parallel Execution",
                "Implement parallel processing capabilities",
                "not_started",
                expected_improvement="40-60% performance improvement"
            ),
            "phase4_advanced": OptimizationPhase(
                "Phase 4: Advanced Optimizations",
                "Intelligent routing and predictive optimization",
                "not_started",
                expected_improvement="50-70% performance improvement"
            )
        }
        
        # Set phase 1 monitoring as completed with current timestamp
        self.phases["phase1_monitoring"].status = "completed"
        self.phases["phase1_monitoring"].start_time = time.time() - 600  # Started 10 minutes ago (example)
        self.phases["phase1_monitoring"].completion_time = time.time() - 300  # Completed 5 minutes ago
        self.phases["phase1_monitoring"].actual_improvement = "Performance baseline established"
        
        # Set phase 1 caching as in progress
        self.phases["phase1_caching"].status = "in_progress"
        self.phases["phase1_caching"].start_time = time.time() - 300  # Started 5 minutes ago
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        performance_stats = performance_monitor.get_system_overview()
        cache_stats = default_schema_cache.get_performance_stats()
        
        # Calculate overall progress
        total_phases = len(self.phases)
        completed_phases = sum(1 for p in self.phases.values() if p.status == "completed")
        in_progress_phases = sum(1 for p in self.phases.values() if p.status == "in_progress")
        
        overall_progress = (completed_phases + (in_progress_phases * 0.5)) / total_phases * 100
        
        # Get current baseline performance
        recent_operations = performance_stats.get("recent_operations", [])
        avg_response_time = 0
        if recent_operations:
            response_times = [op.get("duration", 0) for op in recent_operations if op.get("duration")]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "optimization_progress": {
                "overall_progress_percent": round(overall_progress, 1),
                "completed_phases": completed_phases,
                "in_progress_phases": in_progress_phases,
                "total_phases": total_phases,
                "current_phase": self._get_current_phase()
            },
            "performance_baseline": {
                "average_response_time": round(avg_response_time, 2),
                "total_operations": performance_stats.get("total_operations", 0),
                "success_rate": performance_stats.get("success_rate", 0) * 100,
                "cache_hit_rate": cache_stats.get("hit_rates", {}).get("overall_hit_rate", 0)
            },
            "phase_details": {
                name: {
                    "status": phase.status,
                    "description": phase.description,
                    "duration": round(phase.duration, 2),
                    "expected_improvement": phase.expected_improvement,
                    "actual_improvement": phase.actual_improvement or "In progress"
                }
                for name, phase in self.phases.items()
            },
            "next_steps": self._get_next_steps(),
            "optimization_recommendations": self._get_optimization_recommendations(avg_response_time, cache_stats)
        }
    
    def _get_current_phase(self) -> str:
        """Get the name of the current active phase"""
        for name, phase in self.phases.items():
            if phase.status == "in_progress":
                return phase.phase_name
        
        # If no phase is in progress, return the next one
        for name, phase in self.phases.items():
            if phase.status == "not_started":
                return f"Ready for: {phase.phase_name}"
        
        return "All phases completed"
    
    def _get_next_steps(self) -> List[str]:
        """Get recommended next steps based on current progress"""
        next_steps = []
        
        # Check current phase status
        if self.phases["phase1_caching"].status == "in_progress":
            next_steps.extend([
                "Complete schema cache integration testing",
                "Implement query result caching",
                "Validate Phase 1 performance improvements"
            ])
        elif self.phases["phase1_caching"].status == "completed":
            next_steps.extend([
                "Begin Phase 2: Agent-level optimizations",
                "Implement schema analyst enhancements",
                "Add SQL generator optimization"
            ])
        elif all(p.status == "completed" for p in self.phases.values()):
            next_steps.append("All optimizations completed - monitor and maintain performance")
        
        return next_steps
    
    def _get_optimization_recommendations(self, avg_response_time: float, cache_stats: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on current performance"""
        recommendations = []
        
        # Response time recommendations
        if avg_response_time > 30:
            recommendations.append("High response times detected - prioritize connection pool optimization")
        elif avg_response_time > 20:
            recommendations.append("Moderate response times - consider implementing parallel execution")
        
        # Cache performance recommendations
        hit_rate = cache_stats.get("hit_rates", {}).get("overall_hit_rate", 0)
        if hit_rate < 30:
            recommendations.append("Low cache hit rate - consider prewarming cache with common queries")
        elif hit_rate > 70:
            recommendations.append("Excellent cache performance - ready for advanced optimizations")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performing well - proceed with planned optimization phases")
        
        return recommendations
    
    def complete_phase(self, phase_name: str, actual_improvement: str = ""):
        """Mark a phase as completed"""
        if phase_name in self.phases:
            phase = self.phases[phase_name]
            phase.status = "completed"
            phase.completion_time = time.time()
            if actual_improvement:
                phase.actual_improvement = actual_improvement
    
    def start_phase(self, phase_name: str):
        """Start a new optimization phase"""
        if phase_name in self.phases:
            phase = self.phases[phase_name]
            phase.status = "in_progress"
            phase.start_time = time.time()
    
    def fail_phase(self, phase_name: str, error_message: str = ""):
        """Mark a phase as failed"""
        if phase_name in self.phases:
            phase = self.phases[phase_name]
            phase.status = "failed"
            phase.completion_time = time.time()
            phase.actual_improvement = f"Failed: {error_message}"


# Global optimization tracker instance
optimization_tracker = OptimizationTracker()
