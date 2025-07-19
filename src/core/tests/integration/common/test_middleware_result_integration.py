# ABOUTME: Integration tests for MiddlewareResult execution statistics
# ABOUTME: Tests execution results, performance statistics, status tracking completeness

import asyncio
import time
import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from core.interfaces.middleware import AbstractMiddleware
from core.implementations.memory.middleware.pipeline import InMemoryMiddlewarePipeline
from core.models.middleware import MiddlewareContext, MiddlewareResult, MiddlewareStatus
from core.models.event.event_priority import EventPriority
from core.exceptions.base import TimeoutException, BusinessLogicException


class StatisticsCollectorMiddleware(AbstractMiddleware):
    """Middleware that collects execution statistics."""
    
    def __init__(self, name: str, priority: EventPriority, processing_time_ms: float = 10.0,
                 collect_metrics: bool = True):
        super().__init__(priority)
        self.name = name
        self.processing_time_ms = processing_time_ms
        self.collect_metrics = collect_metrics
        self.execution_history = []
    
    async def process(self, context: MiddlewareContext) -> MiddlewareResult:
        """Process with statistics collection."""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(self.processing_time_ms / 1000.0)
        
        actual_time_ms = (time.time() - start_time) * 1000
        
        # Collect execution metrics
        execution_stats = {
            "start_time": start_time,
            "execution_time_ms": actual_time_ms,
            "context_size": len(context.get_all_data()),
            "metadata_count": len(context.get_all_metadata())
        }
        
        self.execution_history.append(execution_stats)
        
        # Create detailed result with statistics
        result_data = {
            "processed_items": 1,
            "processing_rate": 1000 / actual_time_ms if actual_time_ms > 0 else 0,
            "resource_usage": {
                "memory_mb": 10.5,
                "cpu_percentage": 5.2
            }
        }
        
        result_metadata = {
            "execution_sequence": len(self.execution_history),
            "performance_metrics": execution_stats,
            "middleware_version": "1.0.0"
        }
        
        if self.collect_metrics:
            result_metadata["detailed_metrics"] = {
                "throughput": result_data["processing_rate"],
                "efficiency_score": min(100, 1000 / actual_time_ms),
                "resource_score": 95.0
            }
        
        return MiddlewareResult(
            middleware_name=self.name,
            status=MiddlewareStatus.SUCCESS,
            data=result_data,
            metadata=result_metadata,
            execution_time_ms=actual_time_ms
        )
    
    def can_process(self, context: MiddlewareContext) -> bool:
        return True


class TestMiddlewareResultExecutionStatistics:
    """Test MiddlewareResult execution statistics collection and aggregation."""
    
    @pytest.mark.asyncio
    async def test_execution_result_performance_statistics_status_tracking_complete_record(self):
        """Test complete record: execution results → performance statistics → status tracking completeness."""
        pipeline = InMemoryMiddlewarePipeline("statistics_test")
        
        # Create middleware with different performance characteristics
        middlewares = [
            StatisticsCollectorMiddleware("fast_processor", EventPriority.HIGH, processing_time_ms=5.0),
            StatisticsCollectorMiddleware("medium_processor", EventPriority.NORMAL, processing_time_ms=15.0),
            StatisticsCollectorMiddleware("slow_processor", EventPriority.LOW, processing_time_ms=25.0),
            StatisticsCollectorMiddleware("metrics_collector", EventPriority.VERY_LOW, processing_time_ms=8.0, collect_metrics=True)
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Create test context
        context = MiddlewareContext(
            event_id="statistics_test_001",
            event_type="performance_test",
            user_id="user_stats_001"
        )
        context.set_data("test_data", {"items": 100, "batch_size": 10})
        context.set_metadata("performance_mode", "detailed")
        
        # Execute pipeline and collect statistics
        start_execution = time.time()
        pipeline_result = await pipeline.execute(context)
        total_execution_time = (time.time() - start_execution) * 1000
        
        # Verify pipeline execution results
        assert pipeline_result.status == MiddlewareStatus.SUCCESS
        assert pipeline_result.execution_time_ms is not None
        assert pipeline_result.execution_time_ms > 0
        
        # Verify individual middleware execution statistics
        pipeline_results = pipeline_result.metadata["pipeline_results"]
        assert len(pipeline_results) == 4, "All middleware should have execution results"
        
        # Verify execution order and timing
        execution_times = [result["execution_time_ms"] for result in pipeline_results]
        middleware_names = [result["middleware_name"] for result in pipeline_results]
        
        assert middleware_names == ["fast_processor", "medium_processor", "slow_processor", "metrics_collector"]
        
        # Verify performance characteristics
        assert execution_times[0] < execution_times[1] < execution_times[2], "Execution times should reflect processing complexity"
        
        # Verify detailed statistics for each middleware
        for i, middleware in enumerate(middlewares):
            execution_history = middleware.execution_history
            assert len(execution_history) == 1, f"Middleware {middleware.name} should have one execution record"
            
            stats = execution_history[0]
            assert stats["execution_time_ms"] > 0
            assert stats["context_size"] >= 1  # At least test_data
            assert stats["metadata_count"] >= 1  # At least performance_mode
        
        # Verify pipeline aggregation statistics
        pipeline_data = pipeline_result.data
        assert pipeline_data["total_middlewares"] == 4
        assert pipeline_data["successful_middlewares"] == 4
        assert pipeline_data["failed_middlewares"] == 0
        assert pipeline_data["execution_time_ms"] > 0
        
        # Verify performance metrics collection
        metrics_middleware = middlewares[3]
        metrics_execution = metrics_middleware.execution_history[0]
        
        # Should have detailed metrics in result
        assert "detailed_metrics" in pipeline_results[3]["metadata"]
        detailed_metrics = pipeline_results[3]["metadata"]["detailed_metrics"]
        assert "throughput" in detailed_metrics
        assert "efficiency_score" in detailed_metrics
        assert "resource_score" in detailed_metrics

    @pytest.mark.asyncio
    async def test_middleware_performance_comparison_and_benchmarking(self):
        """Test middleware performance comparison and benchmarking."""
        pipeline = InMemoryMiddlewarePipeline("benchmark_test")
        
        # Create middleware with varying performance profiles
        benchmark_middlewares = [
            StatisticsCollectorMiddleware("baseline", EventPriority.HIGH, processing_time_ms=10.0),
            StatisticsCollectorMiddleware("optimized", EventPriority.HIGH, processing_time_ms=5.0),
            StatisticsCollectorMiddleware("heavyweight", EventPriority.HIGH, processing_time_ms=30.0),
        ]
        
        # Execute each middleware separately for benchmarking
        benchmark_results = {}
        
        for middleware in benchmark_middlewares:
            test_pipeline = InMemoryMiddlewarePipeline(f"benchmark_{middleware.name}")
            await test_pipeline.add_middleware(middleware)
            
            # Create consistent test context
            context = MiddlewareContext(
                event_id=f"benchmark_{middleware.name}",
                event_type="benchmark_test"
            )
            context.set_data("workload_size", 1000)
            
            # Execute multiple times for statistical significance
            execution_times = []
            for run in range(5):
                result = await test_pipeline.execute(context)
                execution_times.append(result.execution_time_ms)
            
            # Calculate benchmark statistics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_dev = (sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5
            
            benchmark_results[middleware.name] = {
                "average_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "std_deviation": std_dev,
                "consistency_score": 100 - (std_dev / avg_time * 100),
                "throughput_ops_per_sec": 1000 / avg_time
            }
        
        # Verify benchmark results
        baseline = benchmark_results["baseline"]
        optimized = benchmark_results["optimized"]
        heavyweight = benchmark_results["heavyweight"]
        
        # Verify performance relationships
        assert optimized["average_time_ms"] < baseline["average_time_ms"], "Optimized should be faster than baseline"
        assert heavyweight["average_time_ms"] > baseline["average_time_ms"], "Heavyweight should be slower than baseline"
        
        # Verify throughput calculations
        assert optimized["throughput_ops_per_sec"] > baseline["throughput_ops_per_sec"]
        assert baseline["throughput_ops_per_sec"] > heavyweight["throughput_ops_per_sec"]
        
        # Verify consistency (lower std_dev = higher consistency)
        for name, result in benchmark_results.items():
            assert result["consistency_score"] > 80, f"Middleware {name} should have consistent performance"

    @pytest.mark.asyncio
    async def test_error_statistics_and_failure_tracking(self):
        """Test error statistics and failure tracking in middleware results."""
        pipeline = InMemoryMiddlewarePipeline("error_stats_test")
        
        class ErrorProneMiddleware(AbstractMiddleware):
            def __init__(self, name: str, priority: EventPriority, failure_rate: float = 0.0):
                super().__init__(priority)
                self.name = name
                self.failure_rate = failure_rate
                self.attempt_count = 0
                self.failure_count = 0
                self.success_count = 0
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                self.attempt_count += 1
                
                # Simulate random failures based on failure rate
                import random
                if random.random() < self.failure_rate:
                    self.failure_count += 1
                    raise BusinessLogicException(
                        f"Simulated failure in {self.name}",
                        "SIM_FAIL_001",
                        {"attempt": self.attempt_count, "failure_count": self.failure_count}
                    )
                
                self.success_count += 1
                
                # Return success result with statistics
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={
                        "attempt": self.attempt_count,
                        "success_count": self.success_count,
                        "failure_count": self.failure_count,
                        "success_rate": self.success_count / self.attempt_count
                    },
                    metadata={
                        "reliability_metrics": {
                            "total_attempts": self.attempt_count,
                            "success_rate": self.success_count / self.attempt_count,
                            "failure_rate": self.failure_count / self.attempt_count
                        }
                    }
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Create middleware with different failure rates
        middlewares = [
            ErrorProneMiddleware("reliable", EventPriority.HIGH, failure_rate=0.0),
            ErrorProneMiddleware("occasional_failure", EventPriority.NORMAL, failure_rate=0.3),
            ErrorProneMiddleware("frequent_failure", EventPriority.LOW, failure_rate=0.7),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Execute pipeline multiple times to collect error statistics
        execution_results = []
        error_statistics = {middleware.name: {"successes": 0, "failures": 0} for middleware in middlewares}
        
        for execution in range(10):
            context = MiddlewareContext(
                event_id=f"error_test_{execution}",
                event_type="error_statistics"
            )
            
            result = await pipeline.execute(context)
            execution_results.append(result)
            
            # Analyze pipeline results for error tracking
            pipeline_results = result.metadata.get("pipeline_results", [])
            
            for middleware_result in pipeline_results:
                middleware_name = middleware_result["middleware_name"]
                status = middleware_result["status"]
                
                if status == "SUCCESS":
                    error_statistics[middleware_name]["successes"] += 1
                elif status == "FAILED":
                    error_statistics[middleware_name]["failures"] += 1
        
        # Verify error statistics match expected failure rates
        for middleware in middlewares:
            stats = error_statistics[middleware.name]
            total_attempts = stats["successes"] + stats["failures"]
            
            if total_attempts > 0:
                actual_failure_rate = stats["failures"] / total_attempts
                expected_failure_rate = middleware.failure_rate
                
                # Allow some tolerance for randomness
                tolerance = 0.2
                assert abs(actual_failure_rate - expected_failure_rate) <= tolerance, \
                    f"Middleware {middleware.name} failure rate {actual_failure_rate} should be close to {expected_failure_rate}"
        
        # Verify error tracking in pipeline aggregation
        total_successes = sum(result.data.get("successful_middlewares", 0) for result in execution_results)
        total_failures = sum(result.data.get("failed_middlewares", 0) for result in execution_results)
        
        assert total_successes > 0, "Should have some successful executions"
        assert total_failures > 0, "Should have some failures due to failure rates"


class TestMiddlewareResultAggregation:
    """Test middleware result aggregation and pipeline-level statistics."""
    
    @pytest.mark.asyncio
    async def test_pipeline_level_statistics_aggregation(self):
        """Test pipeline-level statistics aggregation from individual middleware results."""
        pipeline = InMemoryMiddlewarePipeline("aggregation_test")
        
        # Create middleware that produce different types of results
        class DataProcessingMiddleware(AbstractMiddleware):
            def __init__(self, name: str, priority: EventPriority, 
                         items_processed: int, bytes_processed: int):
                super().__init__(priority)
                self.name = name
                self.items_processed = items_processed
                self.bytes_processed = bytes_processed
            
            async def process(self, context: MiddlewareContext) -> MiddlewareResult:
                processing_time = 0.1  # 100ms
                await asyncio.sleep(processing_time)
                
                return MiddlewareResult(
                    middleware_name=self.name,
                    status=MiddlewareStatus.SUCCESS,
                    data={
                        "items_processed": self.items_processed,
                        "bytes_processed": self.bytes_processed,
                        "processing_rate_items_per_sec": self.items_processed / processing_time,
                        "processing_rate_bytes_per_sec": self.bytes_processed / processing_time
                    },
                    metadata={
                        "performance_tier": "high" if self.items_processed > 100 else "standard",
                        "data_volume": "large" if self.bytes_processed > 1000 else "medium"
                    },
                    execution_time_ms=processing_time * 1000
                )
            
            def can_process(self, context: MiddlewareContext) -> bool:
                return True
        
        # Add middleware with different processing volumes
        middlewares = [
            DataProcessingMiddleware("parser", EventPriority.HIGH, items_processed=50, bytes_processed=2048),
            DataProcessingMiddleware("validator", EventPriority.NORMAL, items_processed=150, bytes_processed=512),
            DataProcessingMiddleware("transformer", EventPriority.LOW, items_processed=75, bytes_processed=1536),
        ]
        
        for middleware in middlewares:
            await pipeline.add_middleware(middleware)
        
        # Execute pipeline
        context = MiddlewareContext(
            event_id="aggregation_test",
            event_type="data_processing"
        )
        
        result = await pipeline.execute(context)
        
        # Verify pipeline-level aggregation
        assert result.status == MiddlewareStatus.SUCCESS
        
        # Verify individual results are captured
        pipeline_results = result.metadata["pipeline_results"]
        assert len(pipeline_results) == 3
        
        # Calculate aggregated statistics
        total_items = sum(r["data"]["items_processed"] for r in pipeline_results)
        total_bytes = sum(r["data"]["bytes_processed"] for r in pipeline_results)
        total_execution_time = sum(r["execution_time_ms"] for r in pipeline_results)
        
        assert total_items == 275  # 50 + 150 + 75
        assert total_bytes == 4096  # 2048 + 512 + 1536
        assert total_execution_time >= 300  # At least 3 * 100ms
        
        # Verify performance tier distribution
        high_tier_count = sum(1 for r in pipeline_results 
                             if r["metadata"]["performance_tier"] == "high")
        standard_tier_count = sum(1 for r in pipeline_results 
                                 if r["metadata"]["performance_tier"] == "standard")
        
        assert high_tier_count == 1  # Only validator processes > 100 items
        assert standard_tier_count == 2  # Parser and transformer
        
        # Verify data volume distribution
        large_volume_count = sum(1 for r in pipeline_results 
                               if r["metadata"]["data_volume"] == "large")
        medium_volume_count = sum(1 for r in pipeline_results 
                                if r["metadata"]["data_volume"] == "medium")
        
        assert large_volume_count == 2  # Parser and transformer > 1000 bytes
        assert medium_volume_count == 1  # Validator <= 1000 bytes


if __name__ == "__main__":
    pytest.main([__file__])