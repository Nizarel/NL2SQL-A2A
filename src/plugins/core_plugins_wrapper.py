"""
Core Plugins Wrapper - TimePlugin and MathPlugin Operations
Provides convenient wrapper methods for TimePlugin and MathPlugin functionality
Similar to MCP Database Plugin pattern but for core Semantic Kernel plugins
"""

import asyncio
from typing import Dict, Any, Optional
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel_pydantic import KernelBaseModel
from pydantic import Field


class CorePluginsWrapper(KernelBaseModel):
    """
    Wrapper for TimePlugin and MathPlugin operations
    Provides convenient methods for time and mathematical operations
    """
    
    kernel: Optional[Kernel] = Field(default=None, exclude=True)
    enable_time_operations: bool = Field(default=True, description="Enable TimePlugin operations")
    enable_math_operations: bool = Field(default=True, description="Enable MathPlugin operations")
    
    def __init__(self, kernel: Kernel = None, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
    
    def set_kernel(self, kernel: Kernel):
        """Set the kernel instance for plugin operations"""
        self.kernel = kernel
    
    # ==================== TIME PLUGIN OPERATIONS ====================
    
    async def get_current_time(self) -> str:
        """Get current time information"""
        if not self.kernel or not self.enable_time_operations:
            raise ValueError("Kernel not set or time operations disabled")
        
        try:
            result = await self.kernel.invoke(plugin_name="time", function_name="now")
            return str(result).strip()
        except Exception as e:
            raise Exception(f"Failed to get current time: {str(e)}")
    
    async def get_date_components(self) -> Dict[str, str]:
        """Get current date components (year, month, day, hour, etc.)"""
        if not self.kernel or not self.enable_time_operations:
            raise ValueError("Kernel not set or time operations disabled")
        
        try:
            components = {}
            
            # Get all date components using correct function names
            components['year'] = str(await self.kernel.invoke(plugin_name="time", function_name="year")).strip()
            components['month'] = str(await self.kernel.invoke(plugin_name="time", function_name="month")).strip()
            components['month_number'] = str(await self.kernel.invoke(plugin_name="time", function_name="month_number")).strip()
            components['day'] = str(await self.kernel.invoke(plugin_name="time", function_name="day")).strip()
            components['hour'] = str(await self.kernel.invoke(plugin_name="time", function_name="hour")).strip()
            components['hour_number'] = str(await self.kernel.invoke(plugin_name="time", function_name="hourNumber")).strip()
            components['minute'] = str(await self.kernel.invoke(plugin_name="time", function_name="minute")).strip()
            components['second'] = str(await self.kernel.invoke(plugin_name="time", function_name="second")).strip()
            components['day_of_week'] = str(await self.kernel.invoke(plugin_name="time", function_name="dayOfWeek")).strip()
            components['time_zone_name'] = str(await self.kernel.invoke(plugin_name="time", function_name="timeZoneName")).strip()
            components['time_zone_offset'] = str(await self.kernel.invoke(plugin_name="time", function_name="timeZoneOffset")).strip()
            
            return components
        except Exception as e:
            raise Exception(f"Failed to get date components: {str(e)}")
    
    async def get_time_zone_info(self) -> Dict[str, str]:
        """Get current timezone information"""
        if not self.kernel or not self.enable_time_operations:
            raise ValueError("Kernel not set or time operations disabled")
        
        try:
            timezone_info = {}
            timezone_info['name'] = str(await self.kernel.invoke(plugin_name="time", function_name="timeZoneName")).strip()
            timezone_info['offset'] = str(await self.kernel.invoke(plugin_name="time", function_name="timeZoneOffset")).strip()
            return timezone_info
        except Exception as e:
            raise Exception(f"Failed to get timezone info: {str(e)}")
    
    async def get_days_ago(self, days: int) -> str:
        """Get date from specified days ago"""
        if not self.kernel or not self.enable_time_operations:
            raise ValueError("Kernel not set or time operations disabled")
        
        try:
            result = await self.kernel.invoke(
                plugin_name="time", 
                function_name="days_ago",
                arguments=KernelArguments(days=str(days))
            )
            return str(result).strip()
        except Exception as e:
            raise Exception(f"Failed to get date {days} days ago: {str(e)}")
    
    async def get_utc_time(self) -> str:
        """Get current UTC time"""
        if not self.kernel or not self.enable_time_operations:
            raise ValueError("Kernel not set or time operations disabled")
        
        try:
            result = await self.kernel.invoke(plugin_name="time", function_name="utcNow")
            return str(result).strip()
        except Exception as e:
            raise Exception(f"Failed to get UTC time: {str(e)}")
    
    # ==================== MATH PLUGIN OPERATIONS ====================
    
    async def add_numbers(self, num1: float, num2: float) -> str:
        """Add two numbers using MathPlugin"""
        if not self.kernel or not self.enable_math_operations:
            raise ValueError("Kernel not set or math operations disabled")
        
        try:
            result = await self.kernel.invoke(
                plugin_name="math",
                function_name="Add",
                arguments=KernelArguments(input=str(num1), amount=str(num2))
            )
            return str(result).strip()
        except Exception as e:
            raise Exception(f"Failed to add {num1} + {num2}: {str(e)}")
    
    async def subtract_numbers(self, num1: float, num2: float) -> str:
        """Subtract two numbers using MathPlugin"""
        if not self.kernel or not self.enable_math_operations:
            raise ValueError("Kernel not set or math operations disabled")
        
        try:
            result = await self.kernel.invoke(
                plugin_name="math",
                function_name="Subtract",
                arguments=KernelArguments(input=str(num1), amount=str(num2))
            )
            return str(result).strip()
        except Exception as e:
            raise Exception(f"Failed to subtract {num1} - {num2}: {str(e)}")
    
    # ==================== BUSINESS ANALYTICS COMBINATIONS ====================
    
    async def calculate_time_based_metrics(self, base_amount: float) -> Dict[str, Any]:
        """Calculate time-based business metrics"""
        if not self.kernel:
            raise ValueError("Kernel not set")
        
        try:
            # Get current time components
            components = await self.get_date_components()
            
            # Calculate various time-based metrics
            metrics = {
                "base_amount": base_amount,
                "current_time": await self.get_current_time(),
                "date_components": components
            }
            
            # Hour-based calculation (use hour_number for numeric value)
            hour_multiplier = float(components['hour_number'])
            hourly_rate = await self.add_numbers(base_amount, hour_multiplier)
            metrics["hourly_rate"] = f"${hourly_rate}"
            
            # Day-based calculation (e.g., daily adjustments)
            day_multiplier = float(components['day'])
            daily_adjustment = await self.subtract_numbers(base_amount, day_multiplier)
            metrics["daily_adjustment"] = f"${daily_adjustment}"
            
            # Month-based calculation (use month_number for numeric value)
            month_multiplier = float(components['month_number'])
            monthly_bonus = await self.add_numbers(base_amount, month_multiplier * 10)
            metrics["monthly_bonus"] = f"${monthly_bonus}"
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Failed to calculate time-based metrics: {str(e)}")
    
    async def get_business_timestamp(self) -> Dict[str, str]:
        """Get comprehensive timestamp for business operations"""
        if not self.kernel:
            raise ValueError("Kernel not set")
        
        try:
            timestamp_info = {
                "local_time": await self.get_current_time(),
                "utc_time": await self.get_utc_time(),
                "components": await self.get_date_components()
            }
            
            # Add formatted business timestamp using numeric values
            components = timestamp_info["components"]
            business_timestamp = f"{components['year']}-{components['month_number'].zfill(2)}-{components['day'].zfill(2)} {components['hour_number'].zfill(2)}:{components['minute'].zfill(2)}:{components['second'].zfill(2)}"
            timestamp_info["business_format"] = business_timestamp
            
            return timestamp_info
            
        except Exception as e:
            raise Exception(f"Failed to get business timestamp: {str(e)}")
    
    async def calculate_financial_projections(self, principal: float, rate: float, periods: int) -> Dict[str, str]:
        """Calculate financial projections using math operations"""
        if not self.kernel or not self.enable_math_operations:
            raise ValueError("Kernel not set or math operations disabled")
        
        try:
            projections = {
                "principal": f"${principal}",
                "rate": f"{rate}%",
                "periods": str(periods)
            }
            
            # Simple interest calculation using add operations
            interest_per_period = principal * (rate / 100)
            total_interest = interest_per_period * periods
            
            # Use MathPlugin for final calculation
            final_amount = await self.add_numbers(principal, total_interest)
            projections["total_interest"] = f"${total_interest}"
            projections["final_amount"] = f"${final_amount}"
            
            # Calculate period-based amounts
            projections["per_period_interest"] = f"${interest_per_period}"
            
            return projections
            
        except Exception as e:
            raise Exception(f"Failed to calculate financial projections: {str(e)}")
    
    # ==================== UTILITY METHODS ====================
    
    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of core plugins"""
        try:
            status = {
                "time_plugin_enabled": self.enable_time_operations,
                "math_plugin_enabled": self.enable_math_operations,
                "kernel_available": self.kernel is not None
            }
            
            if self.kernel:
                # Test time plugin
                try:
                    await self.get_current_time()
                    status["time_plugin_working"] = True
                except:
                    status["time_plugin_working"] = False
                
                # Test math plugin
                try:
                    await self.add_numbers(1, 1)
                    status["math_plugin_working"] = True
                except:
                    status["math_plugin_working"] = False
            else:
                status["time_plugin_working"] = False
                status["math_plugin_working"] = False
            
            return status
            
        except Exception as e:
            return {"error": f"Failed to get plugin status: {str(e)}"}
    
    async def test_all_functions(self) -> Dict[str, Any]:
        """Test all available plugin functions"""
        if not self.kernel:
            raise ValueError("Kernel not set")
        
        results = {
            "time_functions": {},
            "math_functions": {},
            "business_functions": {}
        }
        
        try:
            # Test time functions
            results["time_functions"]["current_time"] = await self.get_current_time()
            results["time_functions"]["date_components"] = await self.get_date_components()
            results["time_functions"]["utc_time"] = await self.get_utc_time()
            
            # Test math functions
            results["math_functions"]["addition"] = await self.add_numbers(100, 25)
            results["math_functions"]["subtraction"] = await self.subtract_numbers(200, 50)
            
            # Test business functions
            results["business_functions"]["time_metrics"] = await self.calculate_time_based_metrics(1000)
            results["business_functions"]["timestamp"] = await self.get_business_timestamp()
            results["business_functions"]["financial"] = await self.calculate_financial_projections(10000, 5, 12)
            
            results["test_status"] = "All tests passed"
            
        except Exception as e:
            results["test_status"] = f"Tests failed: {str(e)}"
        
        return results

    def __str__(self) -> str:
        return f"CorePluginsWrapper(time_enabled={self.enable_time_operations}, math_enabled={self.enable_math_operations}, kernel_set={self.kernel is not None})"
