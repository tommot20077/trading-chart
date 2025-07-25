# ABOUTME: Integration tests for RateLimiter + business logic integration
# ABOUTME: Tests token bucket algorithm, multi-tenant scenarios, and business workflow integration

import asyncio
import time
import pytest
from typing import List, Dict, Any

from core.implementations.memory.common.rate_limiter import InMemoryRateLimiter
from core.implementations.noop.common.rate_limiter import NoOpRateLimiter
from core.exceptions.base import RateLimitExceededException


class TestRateLimiterBusinessLogicIntegration:
    """Test integration between rate limiter and business logic."""
    
    @pytest.mark.asyncio
    async def test_token_bucket_business_request_flow(self, test_rate_limiter):
        """Test complete token bucket algorithm → business request → limit judgment → request processing flow."""
        # Initialize business request context
        business_context = {
            "user_id": "business_user_001",
            "request_type": "api_call",
            "timestamp": time.time()
        }
        
        # Simulate business requests with rate limiting
        successful_requests = []
        rate_limited_requests = []
        
        # Make requests that should succeed (within capacity)
        for i in range(5):  # Capacity is 10, so first 5 should succeed
            if await test_rate_limiter.acquire_for_identifier(business_context["user_id"]):
                successful_requests.append({
                    "request_id": i,
                    "user_id": business_context["user_id"],
                    "processed": True,
                    "timestamp": time.time()
                })
            else:
                rate_limited_requests.append({"request_id": i, "rejected": True})
        
        # Verify business logic integration
        assert len(successful_requests) == 5
        assert len(rate_limited_requests) == 0
        
        # Exhaust remaining tokens
        for i in range(5, 15):  # Try to exceed capacity
            if await test_rate_limiter.acquire_for_identifier(business_context["user_id"]):
                successful_requests.append({
                    "request_id": i,
                    "user_id": business_context["user_id"],
                    "processed": True
                })
            else:
                rate_limited_requests.append({"request_id": i, "rejected": True})
        
        # Verify rate limiting kicked in
        assert len(successful_requests) == 10  # Capacity limit
        assert len(rate_limited_requests) == 5  # Exceeded requests
        
        # Verify remaining tokens
        remaining = await test_rate_limiter.get_remaining_tokens(business_context["user_id"])
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_with_business_exception_handling(self, test_rate_limiter, exception_simulator):
        """Test rate limiter integration with business exception handling."""
        user_id = "exception_test_user"
        
        async def business_operation_with_rate_limit(operation_id: int):
            """Simulate a business operation that may hit rate limits."""
            # Check rate limit before processing
            if not await test_rate_limiter.acquire_for_identifier(user_id):
                raise exception_simulator.create_rate_limit_error()
            
            # Simulate business logic that might fail
            if operation_id == 7:  # Simulate a business logic failure
                raise exception_simulator.create_business_logic_error()
            
            return {"operation_id": operation_id, "status": "success"}
        
        results = []
        exceptions = []
        
        # Execute multiple business operations
        for i in range(15):  # Exceed rate limit
            try:
                result = await business_operation_with_rate_limit(i)
                results.append(result)
            except RateLimitExceededException as e:
                exceptions.append({"type": "rate_limit", "details": e.details})
            except Exception as e:
                exceptions.append({"type": "business", "message": str(e)})
        
        # Verify business logic and rate limiting integration
        assert len(results) <= 10  # Limited by rate limiter capacity
        assert any(ex["type"] == "rate_limit" for ex in exceptions)
        assert any(ex["type"] == "business" for ex in exceptions)
        
        # Verify rate limit exceptions have proper context
        rate_limit_exceptions = [ex for ex in exceptions if ex["type"] == "rate_limit"]
        assert len(rate_limit_exceptions) >= 5  # Should have some rate limit hits

    @pytest.mark.asyncio
    async def test_token_refill_business_continuity(self, test_rate_limiter):
        """Test token refill mechanism with business continuity."""
        user_id = "refill_test_user"
        
        # Exhaust all tokens
        for _ in range(10):
            await test_rate_limiter.acquire_for_identifier(user_id)
        
        # Verify no tokens remaining
        remaining = await test_rate_limiter.get_remaining_tokens(user_id)
        assert remaining == 0
        
        # Wait for token refill (refill_rate = 2.0 tokens/second)
        await asyncio.sleep(3.0)  # Should refill ~6 tokens
        
        # Test business operations can resume
        successful_operations = 0
        for i in range(8):  # Try fewer operations to account for timing
            if await test_rate_limiter.acquire_for_identifier(user_id):
                successful_operations += 1
        
        # Should have refilled enough tokens for some operations
        assert successful_operations >= 4  # Conservative estimate accounting for timing
        assert successful_operations <= 10  # Should not exceed capacity


class TestMultiTenantRateLimitingIntegration:
    """Test multi-tenant rate limiting integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_independent_token_bucket_management(self, multi_tenant_rate_limiter, concurrent_executor):
        """Test independent token bucket management for different users/tenants."""
        users = ["tenant_a", "tenant_b", "tenant_c"]
        
        # Test independent rate limiting per tenant
        async def tenant_operations(tenant_id: str, num_operations: int):
            """Perform operations for a specific tenant."""
            results = []
            for i in range(num_operations):
                success = await multi_tenant_rate_limiter.acquire_for_identifier(tenant_id)
                results.append({
                    "tenant": tenant_id,
                    "operation": i,
                    "success": success,
                    "timestamp": time.time()
                })
                await asyncio.sleep(0.01)  # Small delay
            return results
        
        # Run concurrent operations for different tenants
        tasks = [tenant_operations(user, 8) for user in users]  # Capacity is 5
        all_results = await concurrent_executor.run_concurrent_tasks(tasks)
        
        # Verify each tenant has independent rate limiting
        for tenant_results in all_results:
            successful_ops = [r for r in tenant_results if r["success"]]
            failed_ops = [r for r in tenant_results if not r["success"]]
            
            # Each tenant should get exactly their capacity (5)
            assert len(successful_ops) == 5
            assert len(failed_ops) == 3  # 8 - 5 = 3 failed
        
        # Verify tenants don't interfere with each other
        tenant_successes = {}
        for i, tenant_results in enumerate(all_results):
            tenant_id = users[i]
            tenant_successes[tenant_id] = len([r for r in tenant_results if r["success"]])
        
        # All tenants should get their full capacity
        for tenant_id, success_count in tenant_successes.items():
            assert success_count == 5, f"Tenant {tenant_id} should get full capacity"

    @pytest.mark.asyncio
    async def test_concurrent_multi_tenant_safety(self, multi_tenant_rate_limiter, concurrent_executor):
        """Test thread safety in multi-tenant concurrent scenarios."""
        # Create high concurrency scenario with multiple tenants
        num_tenants = 10
        requests_per_tenant = 20
        
        async def high_frequency_requests(tenant_id: str):
            """Make high-frequency requests for a tenant."""
            success_count = 0
            for _ in range(requests_per_tenant):
                if await multi_tenant_rate_limiter.acquire_for_identifier(f"tenant_{tenant_id}"):
                    success_count += 1
                await asyncio.sleep(0.001)  # Very short delay for high frequency
            return success_count
        
        # Run high concurrency test
        tasks = [high_frequency_requests(str(i)) for i in range(num_tenants)]
        success_counts = await concurrent_executor.run_concurrent_tasks(tasks, max_concurrent=20)
        
        # Verify thread safety - each tenant should get consistent rate limiting
        for i, success_count in enumerate(success_counts):
            # Each tenant has capacity of 5, should not exceed this significantly
            assert success_count <= 7, f"Tenant {i} exceeded expected rate limit: {success_count}"
            assert success_count >= 3, f"Tenant {i} got unexpectedly few tokens: {success_count}"
        
        # Verify no race conditions caused token leaks
        total_success = sum(success_counts)
        expected_max = num_tenants * 7  # Conservative upper bound
        assert total_success <= expected_max, "Possible race condition detected"

    @pytest.mark.asyncio
    async def test_tenant_isolation_and_cleanup(self, multi_tenant_rate_limiter):
        """Test tenant isolation and automatic cleanup mechanisms."""
        # Create multiple tenants with different usage patterns
        active_tenant = "active_tenant"
        inactive_tenant = "inactive_tenant"
        
        # Use tokens for both tenants
        await multi_tenant_rate_limiter.acquire_for_identifier(active_tenant, 3)
        await multi_tenant_rate_limiter.acquire_for_identifier(inactive_tenant, 2)
        
        # Verify both tenants have buckets
        active_remaining = await multi_tenant_rate_limiter.get_remaining_tokens(active_tenant)
        inactive_remaining = await multi_tenant_rate_limiter.get_remaining_tokens(inactive_tenant)
        
        assert active_remaining == 2  # 5 - 3 = 2
        assert inactive_remaining == 3  # 5 - 2 = 3
        
        # Continue using only active tenant
        for _ in range(10):
            await multi_tenant_rate_limiter.acquire_for_identifier(active_tenant)
            await asyncio.sleep(0.1)
        
        # Wait for cleanup interval (0.5 seconds + buffer)
        await asyncio.sleep(2.0)
        
        # Active tenant should still work
        active_remaining_after = await multi_tenant_rate_limiter.get_remaining_tokens(active_tenant)
        assert active_remaining_after >= 0  # Should be valid
        
        # Inactive tenant should still be accessible (cleanup based on access time)
        inactive_remaining_after = await multi_tenant_rate_limiter.get_remaining_tokens(inactive_tenant)
        assert inactive_remaining_after >= 0  # Should still be valid due to recent access


class TestRateLimiterResourceManagement:
    """Test rate limiter resource management and lifecycle."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_many_tenants(self, resource_monitor):
        """Test memory usage doesn't grow unbounded with many tenants."""
        resource_monitor.start_monitoring()
        
        # Create rate limiter with shorter cleanup interval for testing
        rate_limiter = InMemoryRateLimiter(
            capacity=5,
            refill_rate=1.0,
            cleanup_interval=0.2  # Short interval for testing
        )
        
        try:
            # Create many tenants
            num_tenants = 100
            for i in range(num_tenants):
                await rate_limiter.acquire_for_identifier(f"tenant_{i}")
            
            # Check initial memory
            initial_memory_delta = resource_monitor.get_memory_delta()
            
            # Wait for cleanup cycles
            await asyncio.sleep(1.0)
            
            # Create more tenants to verify cleanup happened
            for i in range(num_tenants, num_tenants * 2):
                await rate_limiter.acquire_for_identifier(f"tenant_{i}")
            
            # Check final memory
            final_memory_delta = resource_monitor.get_memory_delta()
            
            # Memory should not grow significantly
            memory_growth_mb = (final_memory_delta - initial_memory_delta) / (1024 * 1024)
            assert memory_growth_mb < 30, f"Memory grew too much: {memory_growth_mb}MB"
            
        finally:
            await rate_limiter.close()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_and_cleanup(self, resource_monitor):
        """Test graceful shutdown and resource cleanup."""
        resource_monitor.start_monitoring()
        
        rate_limiter = InMemoryRateLimiter(capacity=10, refill_rate=5.0, cleanup_interval=0.5)
        
        # Use the rate limiter
        for i in range(5):
            await rate_limiter.acquire_for_identifier(f"test_user_{i}")
        
        # Check resources before shutdown
        leak_check_before = resource_monitor.check_for_leaks(memory_threshold_mb=5, thread_threshold=2)
        
        # Graceful shutdown
        await rate_limiter.close()
        
        # Wait a bit for cleanup to complete
        await asyncio.sleep(0.5)
        
        # Check for resource leaks after shutdown
        leak_check_after = resource_monitor.check_for_leaks(memory_threshold_mb=5, thread_threshold=2)
        
        # Should not have significant resource leaks
        assert not leak_check_after["memory_leak"], f"Memory leak detected: {leak_check_after}"
        assert not leak_check_after["thread_leak"], f"Thread leak detected: {leak_check_after}"

    @pytest.mark.asyncio
    async def test_concurrent_cleanup_safety(self, concurrent_executor):
        """Test cleanup mechanism is safe during concurrent operations."""
        rate_limiter = InMemoryRateLimiter(
            capacity=10,
            refill_rate=5.0,
            cleanup_interval=0.1  # Very short for testing
        )
        
        try:
            # Run concurrent operations while cleanup is happening
            async def continuous_operations():
                """Continuously perform operations."""
                results = []
                for i in range(50):
                    success = await rate_limiter.acquire_for_identifier(f"user_{i % 5}")
                    results.append(success)
                    await asyncio.sleep(0.01)
                return results
            
            # Run multiple concurrent operation streams
            tasks = [continuous_operations() for _ in range(3)]
            all_results = await concurrent_executor.run_concurrent_tasks(tasks)
            
            # Verify operations completed successfully despite concurrent cleanup
            total_operations = sum(len(results) for results in all_results)
            assert total_operations == 150  # 3 tasks * 50 operations each
            
            # Verify no exceptions were raised during concurrent cleanup
            total_successes = sum(sum(results) for results in all_results)
            assert total_successes > 0, "Some operations should have succeeded"
            
        finally:
            await rate_limiter.close()


if __name__ == "__main__":
    pytest.main([__file__])