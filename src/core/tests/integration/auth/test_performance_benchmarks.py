# ABOUTME: Performance benchmark tests for authentication integration
# ABOUTME: Measures and validates performance metrics for auth operations

import pytest
import time
import asyncio
import statistics
from unittest.mock import Mock
from typing import Dict, Any


class TestAuthPerformanceBenchmarks:
    """Performance benchmark tests for authentication operations."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authentication_flow_performance(self, authenticator, authorizer):
        """Benchmark complete authentication flow performance."""
        from core.models.auth.enum import Role, Permission

        # Test 1: Single authentication flow benchmark
        await authenticator.create_user("perf_user", "password123", {Role.USER})

        # Measure login performance
        login_times = []
        for i in range(10):
            start_time = time.perf_counter()
            token_str, auth_token = await authenticator.login("perf_user", "password123")
            end_time = time.perf_counter()
            login_times.append(end_time - start_time)
            await authenticator.logout(token_str)  # Clean up

        # Analyze login performance
        avg_login_time = statistics.mean(login_times)
        max_login_time = max(login_times)
        min_login_time = min(login_times)

        # Performance assertions
        assert avg_login_time < 0.01, f"Average login time {avg_login_time:.4f}s exceeds 10ms threshold"
        assert max_login_time < 0.05, f"Max login time {max_login_time:.4f}s exceeds 50ms threshold"

        # Test 2: Authentication request performance
        token_str, auth_token = await authenticator.login("perf_user", "password123")
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {token_str}"

        auth_times = []
        for i in range(20):
            start_time = time.perf_counter()
            validated_token = await authenticator.authenticate(mock_request)
            end_time = time.perf_counter()
            auth_times.append(end_time - start_time)

        avg_auth_time = statistics.mean(auth_times)
        assert avg_auth_time < 0.005, f"Average auth time {avg_auth_time:.4f}s exceeds 5ms threshold"

        # Test 3: Authorization performance
        authz_times = []
        for i in range(20):
            start_time = time.perf_counter()
            await authorizer.authorize_permission(auth_token, Permission.READ)
            end_time = time.perf_counter()
            authz_times.append(end_time - start_time)

        avg_authz_time = statistics.mean(authz_times)
        assert avg_authz_time < 0.002, f"Average authz time {avg_authz_time:.4f}s exceeds 2ms threshold"

        # Generate performance report
        self._generate_performance_report(
            {
                "login": {
                    "avg": avg_login_time,
                    "min": min_login_time,
                    "max": max_login_time,
                    "samples": len(login_times),
                },
                "authentication": {
                    "avg": avg_auth_time,
                    "min": min(auth_times),
                    "max": max(auth_times),
                    "samples": len(auth_times),
                },
                "authorization": {
                    "avg": avg_authz_time,
                    "min": min(authz_times),
                    "max": max(authz_times),
                    "samples": len(authz_times),
                },
            }
        )

        # Clean up
        await authenticator.delete_user("perf_user")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_authentication_performance(self, authenticator):
        """Benchmark concurrent authentication performance."""
        from core.models.auth.enum import Role

        # Create test users
        user_count = 10
        for i in range(user_count):
            await authenticator.create_user(f"concurrent_user_{i}", "password123", {Role.USER})

        # Test 1: Concurrent login performance
        async def login_user(user_id: int) -> float:
            start_time = time.perf_counter()
            token_str, auth_token = await authenticator.login(f"concurrent_user_{user_id}", "password123")
            end_time = time.perf_counter()
            return end_time - start_time, token_str

        # Measure concurrent logins
        start_time = time.perf_counter()
        tasks = [login_user(i) for i in range(user_count)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        login_times = [result[0] for result in results]
        tokens = [result[1] for result in results]

        # Performance analysis
        avg_concurrent_login = statistics.mean(login_times)
        throughput = user_count / total_time

        assert avg_concurrent_login < 0.02, f"Concurrent login avg {avg_concurrent_login:.4f}s exceeds 20ms"
        assert throughput > 100, f"Login throughput {throughput:.1f} ops/sec below 100 ops/sec threshold"

        # Test 2: Concurrent authentication performance
        mock_requests = []
        for token in tokens:
            mock_req = Mock()
            mock_req.get_header.return_value = f"Bearer {token}"
            mock_requests.append(mock_req)

        async def authenticate_request(mock_request) -> float:
            start_time = time.perf_counter()
            await authenticator.authenticate(mock_request)
            end_time = time.perf_counter()
            return end_time - start_time

        start_time = time.perf_counter()
        auth_tasks = [authenticate_request(req) for req in mock_requests]
        auth_times = await asyncio.gather(*auth_tasks)
        total_auth_time = time.perf_counter() - start_time

        avg_concurrent_auth = statistics.mean(auth_times)
        auth_throughput = len(mock_requests) / total_auth_time

        assert avg_concurrent_auth < 0.01, f"Concurrent auth avg {avg_concurrent_auth:.4f}s exceeds 10ms"
        assert auth_throughput > 200, f"Auth throughput {auth_throughput:.1f} ops/sec below 200 ops/sec"

        # Generate concurrent performance report
        self._generate_concurrent_performance_report(
            {
                "concurrent_login": {
                    "user_count": user_count,
                    "total_time": total_time,
                    "avg_time": avg_concurrent_login,
                    "throughput": throughput,
                },
                "concurrent_auth": {
                    "request_count": len(mock_requests),
                    "total_time": total_auth_time,
                    "avg_time": avg_concurrent_auth,
                    "throughput": auth_throughput,
                },
            }
        )

        # Clean up
        for i in range(user_count):
            await authenticator.delete_user(f"concurrent_user_{i}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_token_lifecycle_performance(self, authenticator):
        """Benchmark token lifecycle operations performance."""
        from core.models.auth.enum import Role

        await authenticator.create_user("lifecycle_user", "password123", {Role.USER})

        # Test 1: Token creation and cleanup performance
        token_creation_times = []
        tokens = []

        # Create multiple tokens
        for i in range(50):
            start_time = time.perf_counter()
            token_str, auth_token = await authenticator.login("lifecycle_user", "password123")
            end_time = time.perf_counter()

            token_creation_times.append(end_time - start_time)
            tokens.append(token_str)

        avg_creation_time = statistics.mean(token_creation_times)
        assert avg_creation_time < 0.01, f"Token creation avg {avg_creation_time:.4f}s exceeds 10ms"

        # Test 2: Token cleanup performance
        start_time = time.perf_counter()
        initial_count = await authenticator.get_token_count()

        # Logout all tokens
        logout_times = []
        for token in tokens:
            logout_start = time.perf_counter()
            await authenticator.logout(token)
            logout_end = time.perf_counter()
            logout_times.append(logout_end - logout_start)

        total_cleanup_time = time.perf_counter() - start_time
        final_count = await authenticator.get_token_count()

        avg_logout_time = statistics.mean(logout_times)
        cleanup_throughput = len(tokens) / total_cleanup_time

        assert avg_logout_time < 0.005, f"Token logout avg {avg_logout_time:.4f}s exceeds 5ms"
        assert cleanup_throughput > 500, f"Cleanup throughput {cleanup_throughput:.1f} ops/sec below 500"

        # Test 3: Expired token cleanup performance
        # Create tokens with short TTL
        original_ttl = authenticator.default_token_ttl
        authenticator.default_token_ttl = 1  # 1 second

        try:
            # Create short-lived tokens
            short_tokens = []
            for i in range(20):
                token_str, _ = await authenticator.login("lifecycle_user", "password123")
                short_tokens.append(token_str)

            # Wait for expiry
            await asyncio.sleep(1.5)

            # Measure cleanup performance
            start_time = time.perf_counter()
            expired_count = await authenticator.cleanup_expired_tokens()
            cleanup_time = time.perf_counter() - start_time

            assert cleanup_time < 0.1, f"Expired token cleanup {cleanup_time:.4f}s exceeds 100ms"
            assert expired_count >= 20, f"Expected to clean up 20+ tokens, cleaned {expired_count}"

        finally:
            authenticator.default_token_ttl = original_ttl

        # Generate lifecycle performance report
        self._generate_lifecycle_performance_report(
            {
                "token_creation": {
                    "count": len(tokens),
                    "avg_time": avg_creation_time,
                    "total_time": sum(token_creation_times),
                },
                "token_cleanup": {"count": len(tokens), "avg_time": avg_logout_time, "throughput": cleanup_throughput},
                "expired_cleanup": {"cleanup_time": cleanup_time, "expired_count": expired_count},
            }
        )

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
