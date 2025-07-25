# ABOUTME: Performance benchmark tests for authentication integration
# ABOUTME: Measures and validates performance metrics for auth operations

import pytest
import asyncio
from unittest.mock import Mock
from typing import Dict, Any


class TestAuthPerformanceBenchmarks:
    """Performance benchmark tests for authentication operations."""

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authentication_flow_performance(self, authenticator, authorizer, benchmark):
        """Benchmark complete authentication flow performance."""
        from core.models.auth.enum import Role

        # Setup user
        await authenticator.create_user("perf_user", "password123", {Role.USER})

        async def auth_flow_operation():
            # Login and logout cycle
            token_str, auth_token = await authenticator.login("perf_user", "password123")
            await authenticator.logout(token_str)
            return True

        # Benchmark login flow
        result = await benchmark(auth_flow_operation)
        assert result is True

        # Clean up
        await authenticator.delete_user("perf_user")

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_authentication_performance(self, authenticator, benchmark):
        """Benchmark concurrent authentication performance."""
        from core.models.auth.enum import Role

        # Create test users
        user_count = 5  # Reduced for benchmark consistency
        for i in range(user_count):
            await authenticator.create_user(f"concurrent_user_{i}", "password123", {Role.USER})

        async def concurrent_operations():
            # Concurrent login operations
            async def login_user(user_id: int):
                token_str, auth_token = await authenticator.login(f"concurrent_user_{user_id}", "password123")
                return token_str

            tasks = [login_user(i) for i in range(user_count)]
            tokens = await asyncio.gather(*tasks)

            # Concurrent authentication operations
            mock_requests = []
            for token in tokens:
                mock_req = Mock()
                mock_req.get_header.return_value = f"Bearer {token}"
                mock_requests.append(mock_req)

            auth_tasks = [authenticator.authenticate(req) for req in mock_requests]
            await asyncio.gather(*auth_tasks)

            return len(tokens) + len(mock_requests)  # Total operations

        # Benchmark concurrent operations
        result = await benchmark(concurrent_operations)
        assert result == user_count * 2  # Login + auth operations

        # Clean up
        for i in range(user_count):
            await authenticator.delete_user(f"concurrent_user_{i}")

    @pytest.mark.benchmark
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_lifecycle_performance(self, authenticator, benchmark):
        """Benchmark token lifecycle operations performance."""
        from core.models.auth.enum import Role

        await authenticator.create_user("lifecycle_user", "password123", {Role.USER})

        async def token_lifecycle_operations():
            # Token creation
            tokens = []
            for i in range(25):  # Reduced for benchmark
                token_str, auth_token = await authenticator.login("lifecycle_user", "password123")
                tokens.append(token_str)

            # Token cleanup
            for token in tokens:
                await authenticator.logout(token)

            # Expired token cleanup test
            original_ttl = getattr(authenticator, "default_token_ttl", None)
            if hasattr(authenticator, "default_token_ttl"):
                authenticator.default_token_ttl = 1

                # Create short-lived tokens
                short_tokens = []
                for i in range(10):
                    token_str, _ = await authenticator.login("lifecycle_user", "password123")
                    short_tokens.append(token_str)

                # Wait for expiry
                await asyncio.sleep(1.5)

                # Cleanup expired tokens
                if hasattr(authenticator, "cleanup_expired_tokens"):
                    expired_count = await authenticator.cleanup_expired_tokens()
                else:
                    expired_count = 0

                # Restore original TTL
                if original_ttl is not None:
                    authenticator.default_token_ttl = original_ttl

                return len(tokens) + len(short_tokens) + expired_count
            else:
                return len(tokens)

        # Benchmark token lifecycle
        result = await benchmark(token_lifecycle_operations)
        assert result > 0

        # Clean up
        await authenticator.delete_user("lifecycle_user")

    def _generate_performance_report(self, metrics: Dict[str, Any]):
        """Generate performance benchmark report."""
        from pathlib import Path

        report_content = []
        report_content.append("# Authentication Performance Benchmark Report")
        report_content.append("")

        for operation, data in metrics.items():
            report_content.append(f"## {operation.title()} Performance")
            report_content.append(f"- Average Time: {data['avg'] * 1000:.2f}ms")
            report_content.append(f"- Min Time: {data['min'] * 1000:.2f}ms")
            report_content.append(f"- Max Time: {data['max'] * 1000:.2f}ms")
            report_content.append(f"- Samples: {data['samples']}")
            report_content.append("")

        # Performance thresholds
        report_content.append("## Performance Thresholds")
        report_content.append("- Login: < 10ms average")
        report_content.append("- Authentication: < 5ms average")
        report_content.append("- Authorization: < 2ms average")
        report_content.append("")

        # Save report
        report_file = Path(__file__).parent / "performance_benchmark_report.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_content))

    def _generate_concurrent_performance_report(self, metrics: Dict[str, Any]):
        """Generate concurrent performance report."""
        from pathlib import Path

        report_content = []
        report_content.append("# Concurrent Authentication Performance Report")
        report_content.append("")

        for operation, data in metrics.items():
            report_content.append(f"## {operation.replace('_', ' ').title()}")
            if "user_count" in data:
                report_content.append(f"- Users: {data['user_count']}")
            if "request_count" in data:
                report_content.append(f"- Requests: {data['request_count']}")
            report_content.append(f"- Total Time: {data['total_time'] * 1000:.2f}ms")
            report_content.append(f"- Average Time: {data['avg_time'] * 1000:.2f}ms")
            report_content.append(f"- Throughput: {data['throughput']:.1f} ops/sec")
            report_content.append("")

        # Save report
        report_file = Path(__file__).parent / "concurrent_performance_report.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_content))

    def _generate_lifecycle_performance_report(self, metrics: Dict[str, Any]):
        """Generate token lifecycle performance report."""
        from pathlib import Path

        report_content = []
        report_content.append("# Token Lifecycle Performance Report")
        report_content.append("")

        # Token creation
        creation = metrics["token_creation"]
        report_content.append("## Token Creation Performance")
        report_content.append(f"- Tokens Created: {creation['count']}")
        report_content.append(f"- Average Time: {creation['avg_time'] * 1000:.2f}ms")
        report_content.append(f"- Total Time: {creation['total_time'] * 1000:.2f}ms")
        report_content.append("")

        # Token cleanup
        cleanup = metrics["token_cleanup"]
        report_content.append("## Token Cleanup Performance")
        report_content.append(f"- Tokens Cleaned: {cleanup['count']}")
        report_content.append(f"- Average Time: {cleanup['avg_time'] * 1000:.2f}ms")
        report_content.append(f"- Throughput: {cleanup['throughput']:.1f} ops/sec")
        report_content.append("")

        # Expired cleanup
        expired = metrics["expired_cleanup"]
        report_content.append("## Expired Token Cleanup Performance")
        report_content.append(f"- Cleanup Time: {expired['cleanup_time'] * 1000:.2f}ms")
        report_content.append(f"- Tokens Cleaned: {expired['expired_count']}")
        report_content.append("")

        # Save report
        report_file = Path(__file__).parent / "lifecycle_performance_report.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_content))
