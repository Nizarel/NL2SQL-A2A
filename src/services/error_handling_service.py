"""
Error Handling Service - Advanced standardized error handling across all agents
Provides consistent error formatting, logging, categorization, and response patterns
"""

import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from enum import Enum


class ErrorCategory(Enum):
    """Comprehensive error categories for better classification"""
    SQL_SYNTAX = "sql_syntax"
    SQL_EXECUTION = "sql_execution"
    DATABASE_CONNECTION = "database_connection"
    SCHEMA_ANALYSIS = "schema_analysis"
    TEMPLATE_PROCESSING = "template_processing"
    AGENT_COMMUNICATION = "agent_communication"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    DATA_FORMAT = "data_format"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    MEMORY = "memory"
    GENERAL = "general"


class ErrorSeverity(Enum):
    """Error severity levels for proper escalation"""
    CRITICAL = "critical"      # System failure, immediate attention
    HIGH = "high"             # Major functionality affected
    MEDIUM = "medium"         # Minor functionality affected  
    LOW = "low"               # Cosmetic or minor issues
    INFO = "info"             # Informational messages


class ErrorHandlingService:
    """
    Advanced centralized service for error handling operations including:
    - Comprehensive error categorization
    - Severity-based error classification
    - Context-aware error recovery
    - Intelligent suggestion generation
    - Performance impact assessment
    """
    
    @staticmethod
    def create_error_response(
        error_message: str,
        error_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized error response format
        
        Args:
            error_message: Primary error description
            error_type: Category of error (sql, validation, execution, etc.)
            context: Additional context information
            suggestions: Recovery or troubleshooting suggestions
            
        Returns:
            Standardized error response dictionary
        """
        return {
            "success": False,
            "error": error_message,
            "error_type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context or {},
            "suggestions": suggestions or [],
            "data": None,
            "metadata": {
                "error_occurred": True,
                "error_category": error_type,
                "recovery_suggestions": suggestions or []
            }
        }
    
    @staticmethod
    def handle_sql_error(
        error: Exception,
        sql_query: Optional[str] = None,
        operation: str = "execution"
    ) -> Dict[str, Any]:
        """
        Handle SQL-specific errors with enhanced context
        
        Args:
            error: The caught exception
            sql_query: SQL query that caused the error
            operation: Type of SQL operation (generation, validation, execution)
            
        Returns:
            Enhanced error response with SQL-specific guidance
        """
        error_message = str(error)
        suggestions = []
        
        # Provide helpful error messages for common SQL Server issues
        if "Incorrect syntax near '12 months'" in error_message:
            enhanced_message = "SQL Server syntax error: Use DATEADD(MONTH, -12, GETDATE()) instead of INTERVAL '12 months'"
            suggestions.append("Replace INTERVAL syntax with DATEADD function")
            suggestions.append("Use SQL Server-specific date functions")
            
        elif "Incorrect syntax near 'interval'" in error_message.lower():
            enhanced_message = "SQL Server syntax error: INTERVAL is not supported. Use DATEADD() for date arithmetic"
            suggestions.append("Convert INTERVAL expressions to DATEADD functions")
            suggestions.append("Review SQL Server date arithmetic documentation")
            
        elif "Invalid object name" in error_message and "dev." not in error_message:
            enhanced_message = f"Table not found: Ensure table names use 'dev.' schema prefix. Original error: {error_message}"
            suggestions.append("Add 'dev.' prefix to all table names")
            suggestions.append("Verify table names match the database schema")
            
        elif "Incorrect syntax" in error_message:
            enhanced_message = f"SQL syntax error: {error_message}. Check for SQL Server compatibility issues"
            suggestions.append("Review query for SQL Server syntax compliance")
            suggestions.append("Check for PostgreSQL/MySQL syntax that needs conversion")
            
        else:
            enhanced_message = f"Database {operation} error: {error_message}"
            suggestions.append(f"Review {operation} operation for potential issues")
        
        context = {
            "operation": operation,
            "original_error": error_message,
            "error_class": type(error).__name__
        }
        
        if sql_query:
            context["sql_query"] = sql_query
            context["query_length"] = len(sql_query)
        
        return ErrorHandlingService.create_error_response(
            error_message=enhanced_message,
            error_type="sql_error",
            context=context,
            suggestions=suggestions
        )
    
    @staticmethod
    def handle_validation_error(
        validation_result: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle validation errors with specific guidance
        
        Args:
            validation_result: Result from validation operation
            input_data: Original input that failed validation
            
        Returns:
            Enhanced validation error response
        """
        error_message = validation_result.get("error", "Validation failed")
        
        suggestions = [
            "Review input data format and requirements",
            "Check for missing required fields",
            "Validate data types and constraints"
        ]
        
        # Add specific suggestions based on validation type
        if "empty" in error_message.lower():
            suggestions.insert(0, "Provide non-empty input data")
        elif "sql" in error_message.lower():
            suggestions.insert(0, "Ensure SQL query starts with SELECT or WITH")
            suggestions.append("Verify SQL syntax follows SQL Server standards")
        
        context = {
            "validation_details": validation_result,
            "operation": "validation"
        }
        
        if input_data:
            context["input_summary"] = {
                "keys": list(input_data.keys()) if isinstance(input_data, dict) else None,
                "data_type": type(input_data).__name__
            }
        
        return ErrorHandlingService.create_error_response(
            error_message=error_message,
            error_type="validation_error",
            context=context,
            suggestions=suggestions
        )
    
    @staticmethod
    def handle_agent_processing_error(
        error: Exception,
        agent_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        step: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle agent processing errors with context
        
        Args:
            error: The caught exception
            agent_name: Name of the agent that encountered the error
            input_data: Input data being processed
            step: Current processing step when error occurred
            
        Returns:
            Enhanced agent error response
        """
        error_message = str(error)
        
        suggestions = [
            f"Review {agent_name} configuration and dependencies",
            "Check input data format and completeness",
            "Verify system resources and connectivity"
        ]
        
        # Add agent-specific suggestions
        if "sql" in agent_name.lower():
            suggestions.extend([
                "Verify database connectivity and schema access",
                "Check SQL generation templates and configuration"
            ])
        elif "orchestrator" in agent_name.lower():
            suggestions.extend([
                "Review agent workflow configuration",
                "Check agent dependencies and initialization"
            ])
        elif "executor" in agent_name.lower():
            suggestions.extend([
                "Verify database connection and permissions",
                "Check query execution limits and timeouts"
            ])
        
        context = {
            "agent": agent_name,
            "processing_step": step,
            "error_class": type(error).__name__,
            "operation": "agent_processing"
        }
        
        if input_data:
            context["input_summary"] = {
                "keys": list(input_data.keys()) if isinstance(input_data, dict) else None,
                "data_size": len(str(input_data))
            }
        
        return ErrorHandlingService.create_error_response(
            error_message=f"{agent_name} processing failed: {error_message}",
            error_type="agent_error",
            context=context,
            suggestions=suggestions
        )
    
    @staticmethod
    def handle_api_error(
        error: Exception,
        endpoint: str,
        request_data: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ) -> Dict[str, Any]:
        """
        Handle API endpoint errors
        
        Args:
            error: The caught exception
            endpoint: API endpoint that encountered the error
            request_data: Request data that caused the error
            status_code: HTTP status code
            
        Returns:
            Enhanced API error response
        """
        error_message = str(error)
        
        suggestions = [
            "Check request format and required parameters",
            "Verify API endpoint availability",
            "Review authentication and permissions"
        ]
        
        if status_code == 400:
            suggestions.insert(0, "Validate request data format and required fields")
        elif status_code == 401:
            suggestions.insert(0, "Check authentication credentials")
        elif status_code == 403:
            suggestions.insert(0, "Verify user permissions for this operation")
        elif status_code == 404:
            suggestions.insert(0, "Check endpoint URL and availability")
        elif status_code >= 500:
            suggestions.insert(0, "Server error - check system status and retry")
        
        context = {
            "endpoint": endpoint,
            "http_status": status_code,
            "error_class": type(error).__name__,
            "operation": "api_request"
        }
        
        if request_data:
            context["request_summary"] = {
                "keys": list(request_data.keys()) if isinstance(request_data, dict) else None,
                "data_size": len(str(request_data))
            }
        
        return ErrorHandlingService.create_error_response(
            error_message=f"API error at {endpoint}: {error_message}",
            error_type="api_error",
            context=context,
            suggestions=suggestions
        )
    
    @staticmethod
    def log_error(
        error_response: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Log error with consistent format
        
        Args:
            error_response: Error response from other methods
            logger: Logger instance to use
            correlation_id: Request correlation ID for tracking
        """
        if not logger:
            logger = logging.getLogger(__name__)
        
        error_context = {
            "error_type": error_response.get("error_type", "unknown"),
            "timestamp": error_response.get("timestamp"),
            "correlation_id": correlation_id,
            "context": error_response.get("context", {})
        }
        
        logger.error(
            f"Error occurred: {error_response.get('error', 'Unknown error')}",
            extra=error_context
        )
    
    @staticmethod
    def create_success_response(
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create standardized success response format
        
        Args:
            data: Response data
            metadata: Additional metadata
            message: Optional success message
            
        Returns:
            Standardized success response dictionary
        """
        return {
            "success": True,
            "data": data,
            "error": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "metadata": metadata or {}
        }
    
    @staticmethod
    def wrap_agent_response(
        agent_method,
        agent_name: str,
        input_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Wrapper for agent methods to provide consistent error handling
        
        Args:
            agent_method: Agent method to call
            agent_name: Name of the agent
            input_data: Input data for the method
            correlation_id: Request correlation ID
            
        Returns:
            Standardized response (success or error)
        """
        try:
            result = agent_method(input_data)
            
            # If result is already a standardized response, return it
            if isinstance(result, dict) and "success" in result:
                return result
            
            # Otherwise, wrap in success response
            return ErrorHandlingService.create_success_response(
                data=result,
                metadata={"agent": agent_name, "correlation_id": correlation_id}
            )
            
        except Exception as e:
            error_response = ErrorHandlingService.handle_agent_processing_error(
                error=e,
                agent_name=agent_name,
                input_data=input_data
            )
            
            # Log the error
            ErrorHandlingService.log_error(
                error_response=error_response,
                correlation_id=correlation_id
            )
            
            return error_response
    
    @staticmethod
    def categorize_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """
        Intelligently categorize errors based on type and context
        
        Args:
            error: The exception that occurred
            context: Additional context for categorization
            
        Returns:
            Appropriate error category
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # SQL-related errors
        if any(keyword in error_str for keyword in ['syntax', 'sql', 'query', 'column', 'table']):
            if 'syntax' in error_str or 'invalid' in error_str:
                return ErrorCategory.SQL_SYNTAX
            else:
                return ErrorCategory.SQL_EXECUTION
        
        # Connection and network errors
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network', 'unreachable']):
            if 'timeout' in error_str:
                return ErrorCategory.TIMEOUT
            else:
                return ErrorCategory.DATABASE_CONNECTION
        
        # Authentication errors
        if any(keyword in error_str for keyword in ['auth', 'permission', 'forbidden', 'unauthorized']):
            return ErrorCategory.AUTHENTICATION
        
        # Memory and resource errors
        if any(keyword in error_str for keyword in ['memory', 'limit', 'quota', 'resource']):
            return ErrorCategory.RESOURCE_LIMIT
        
        # Template and schema errors
        if any(keyword in error_str for keyword in ['template', 'jinja', 'render']):
            return ErrorCategory.TEMPLATE_PROCESSING
        if any(keyword in error_str for keyword in ['schema', 'metadata']):
            return ErrorCategory.SCHEMA_ANALYSIS
        
        # Configuration errors
        if any(keyword in error_str for keyword in ['config', 'setting', 'env', 'variable']):
            return ErrorCategory.CONFIGURATION
        
        # Validation errors
        if any(keyword in error_str for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION
        
        return ErrorCategory.GENERAL
    
    @staticmethod
    def determine_severity(
        error: Exception, 
        category: ErrorCategory, 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorSeverity:
        """
        Determine error severity based on category and impact
        
        Args:
            error: The exception that occurred
            category: Error category
            context: Additional context for severity determination
            
        Returns:
            Appropriate error severity level
        """
        # Critical errors that break core functionality
        critical_categories = [
            ErrorCategory.DATABASE_CONNECTION,
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.MEMORY
        ]
        
        # High severity errors that impact major features
        high_categories = [
            ErrorCategory.SQL_EXECUTION,
            ErrorCategory.AGENT_COMMUNICATION,
            ErrorCategory.CONFIGURATION
        ]
        
        # Medium severity errors that impact specific features
        medium_categories = [
            ErrorCategory.SQL_SYNTAX,
            ErrorCategory.SCHEMA_ANALYSIS,
            ErrorCategory.TEMPLATE_PROCESSING,
            ErrorCategory.TIMEOUT
        ]
        
        if category in critical_categories:
            return ErrorSeverity.CRITICAL
        elif category in high_categories:
            return ErrorSeverity.HIGH
        elif category in medium_categories:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    @staticmethod
    def generate_intelligent_suggestions(
        error: Exception,
        category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate intelligent recovery suggestions based on error analysis
        
        Args:
            error: The exception that occurred
            category: Error category
            context: Additional context for suggestion generation
            
        Returns:
            List of actionable recovery suggestions
        """
        suggestions = []
        error_str = str(error).lower()
        
        if category == ErrorCategory.SQL_SYNTAX:
            suggestions.extend([
                "Check SQL syntax for missing commas, quotes, or parentheses",
                "Verify table and column names are spelled correctly",
                "Ensure proper JOIN syntax and conditions",
                "Validate WHERE clause formatting"
            ])
        
        elif category == ErrorCategory.SQL_EXECUTION:
            suggestions.extend([
                "Verify database connection is active",
                "Check if referenced tables and columns exist",
                "Ensure user has appropriate permissions",
                "Review query complexity and timeout settings"
            ])
        
        elif category == ErrorCategory.DATABASE_CONNECTION:
            suggestions.extend([
                "Verify database server is running and accessible",
                "Check network connectivity to database",
                "Validate connection string parameters",
                "Ensure firewall rules allow database access"
            ])
        
        elif category == ErrorCategory.TEMPLATE_PROCESSING:
            suggestions.extend([
                "Check template file exists and is readable",
                "Verify template syntax is valid Jinja2",
                "Ensure all required template variables are provided",
                "Check for circular template includes"
            ])
        
        elif category == ErrorCategory.AUTHENTICATION:
            suggestions.extend([
                "Verify API keys and credentials are correct",
                "Check if credentials have expired",
                "Ensure user has appropriate permissions",
                "Validate authentication configuration"
            ])
        
        elif category == ErrorCategory.TIMEOUT:
            suggestions.extend([
                "Increase timeout settings for complex operations",
                "Optimize query performance to reduce execution time",
                "Check network latency and connection stability",
                "Consider breaking large operations into smaller chunks"
            ])
        
        elif category == ErrorCategory.RESOURCE_LIMIT:
            suggestions.extend([
                "Check available memory and disk space",
                "Review resource quotas and limits",
                "Optimize query to reduce resource usage",
                "Consider scaling up resources if needed"
            ])
        
        else:
            suggestions.extend([
                "Review error logs for detailed information",
                "Check system configuration and settings",
                "Verify all dependencies are properly installed",
                "Contact support if issue persists"
            ])
        
        return suggestions
    
    @staticmethod
    def create_enhanced_error_response(
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive error response with intelligent analysis
        
        Args:
            error: The exception that occurred
            context: Additional context information
            correlation_id: Request correlation ID
            
        Returns:
            Enhanced error response with categorization and suggestions
        """
        category = ErrorHandlingService.categorize_error(error, context)
        severity = ErrorHandlingService.determine_severity(error, category, context)
        suggestions = ErrorHandlingService.generate_intelligent_suggestions(error, category, context)
        
        return {
            "success": False,
            "error": str(error),
            "error_type": category.value,
            "severity": severity.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "context": context or {},
            "suggestions": suggestions,
            "data": None,
            "metadata": {
                "error_occurred": True,
                "error_category": category.value,
                "error_severity": severity.value,
                "recovery_suggestions": suggestions,
                "error_class": type(error).__name__,
                "analysis": {
                    "auto_categorized": True,
                    "severity_determined": True,
                    "suggestions_generated": len(suggestions)
                }
            }
        }
    
    @staticmethod
    def assess_performance_impact(
        error: Exception,
        category: ErrorCategory,
        processing_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Assess the performance impact of an error
        
        Args:
            error: The exception that occurred
            category: Error category
            processing_time: Time taken before error occurred
            
        Returns:
            Performance impact assessment
        """
        impact_level = "low"
        impact_description = "Minimal performance impact"
        
        # High impact categories
        if category in [ErrorCategory.DATABASE_CONNECTION, ErrorCategory.MEMORY, ErrorCategory.RESOURCE_LIMIT]:
            impact_level = "high"
            impact_description = "Significant performance degradation expected"
        
        # Medium impact categories
        elif category in [ErrorCategory.SQL_EXECUTION, ErrorCategory.TIMEOUT, ErrorCategory.NETWORK]:
            impact_level = "medium"
            impact_description = "Moderate performance impact on specific operations"
        
        return {
            "impact_level": impact_level,
            "description": impact_description,
            "processing_time": processing_time,
            "category": category.value,
            "recommendations": {
                "monitor": impact_level in ["high", "medium"],
                "alert": impact_level == "high",
                "investigate": impact_level == "high"
            }
        }
