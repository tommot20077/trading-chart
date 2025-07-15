# ABOUTME: Test stability verification for authentication integration tests
# ABOUTME: Validates test reliability and consistency across multiple runs

import pytest
import asyncio
import statistics
import time
import time_machine
from typing import Dict, Any
from pathlib import Path


class TestStabilityVerification:
    """Test stability verification for authentication integration tests."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_authentication_flow_stability(self, authenticator, authorizer):
        """Verify authentication flow stability across multiple runs."""
        from core.models.auth.enum import Role, Permission
        from unittest.mock import Mock

        # Test configuration
        test_runs = 5
        operations_per_run = 10

        # Create test user
        await authenticator.create_user("stability_user", "password123", {Role.USER})

        # Test 1: Login stability
        login_success_rates = []
        login_times = []

        for run in range(test_runs):
            successes = 0
            run_times = []

            for i in range(operations_per_run):
                try:
                    start_time = time.perf_counter()
                    token_str, auth_token = await authenticator.login("stability_user", "password123")
                    end_time = time.perf_counter()

                    successes += 1
                    run_times.append(end_time - start_time)

                    # Clean up immediately
                    await authenticator.logout(token_str)

                except Exception as e:
                    print(f"Login failed in run {run}, operation {i}: {e}")

            success_rate = successes / operations_per_run
            login_success_rates.append(success_rate)
            login_times.extend(run_times)

        # Analyze login stability
        avg_success_rate = statistics.mean(login_success_rates)
        min_success_rate = min(login_success_rates)
        time_variance = statistics.variance(login_times) if len(login_times) > 1 else 0

        assert avg_success_rate >= 0.99, f"Login success rate {avg_success_rate:.3f} below 99%"
        assert min_success_rate >= 0.95, f"Min success rate {min_success_rate:.3f} below 95%"
        assert time_variance < 0.001, f"Login time variance {time_variance:.6f} too high"

        # Test 2: Authentication stability
        token_str, auth_token = await authenticator.login("stability_user", "password123")
        mock_request = Mock()
        mock_request.get_header.return_value = f"Bearer {token_str}"

        auth_success_rates = []
        auth_times = []

        for run in range(test_runs):
            successes = 0
            run_times = []

            for i in range(operations_per_run):
                try:
                    start_time = time.perf_counter()
                    validated_token = await authenticator.authenticate(mock_request)
                    end_time = time.perf_counter()

                    assert validated_token.username == "stability_user"
                    successes += 1
                    run_times.append(end_time - start_time)

                except Exception as e:
                    print(f"Auth failed in run {run}, operation {i}: {e}")

            success_rate = successes / operations_per_run
            auth_success_rates.append(success_rate)
            auth_times.extend(run_times)

        # Analyze authentication stability
        avg_auth_success = statistics.mean(auth_success_rates)
        min_auth_success = min(auth_success_rates)
        auth_time_variance = statistics.variance(auth_times) if len(auth_times) > 1 else 0

        assert avg_auth_success >= 0.99, f"Auth success rate {avg_auth_success:.3f} below 99%"
        assert min_auth_success >= 0.98, f"Min auth success rate {min_auth_success:.3f} below 98%"

        # Test 3: Authorization stability
        authz_success_rates = []
        authz_times = []

        for run in range(test_runs):
            successes = 0
            run_times = []

            for i in range(operations_per_run):
                try:
                    start_time = time.perf_counter()
                    await authorizer.authorize_permission(auth_token, Permission.READ)
                    end_time = time.perf_counter()

                    successes += 1
                    run_times.append(end_time - start_time)

                except Exception as e:
                    print(f"Authz failed in run {run}, operation {i}: {e}")

            success_rate = successes / operations_per_run
            authz_success_rates.append(success_rate)
            authz_times.extend(run_times)

        # Analyze authorization stability
        avg_authz_success = statistics.mean(authz_success_rates)
        min_authz_success = min(authz_success_rates)

        assert avg_authz_success >= 0.99, f"Authz success rate {avg_authz_success:.3f} below 99%"
        assert min_authz_success >= 0.98, f"Min authz success rate {min_authz_success:.3f} below 98%"

        # Generate stability report
        self._generate_stability_report(
            {
                "login": {
                    "avg_success_rate": avg_success_rate,
                    "min_success_rate": min_success_rate,
                    "time_variance": time_variance,
                    "total_operations": len(login_times),
                },
                "authentication": {
                    "avg_success_rate": avg_auth_success,
                    "min_success_rate": min_auth_success,
                    "time_variance": auth_time_variance,
                    "total_operations": len(auth_times),
                },
                "authorization": {
                    "avg_success_rate": avg_authz_success,
                    "min_success_rate": min_authz_success,
                    "total_operations": len(authz_times),
                },
            }
        )

        # Clean up
        await authenticator.logout(token_str)
        await authenticator.delete_user("stability_user")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_operation_stability(self, authenticator):
        """Verify stability under concurrent operations."""
        from core.models.auth.enum import Role

        # Test configuration
        concurrent_users = 5
        operations_per_user = 10
        test_rounds = 3

        # Create test users
        for i in range(concurrent_users):
            await authenticator.create_user(f"concurrent_user_{i}", "password123", {Role.USER})

        stability_results = []

        for round_num in range(test_rounds):
            # Test concurrent login/logout cycles
            async def user_operations(user_id: int) -> Dict[str, Any]:
                successes = 0
                failures = 0

                for op in range(operations_per_user):
                    try:
                        # Login
                        token_str, auth_token = await authenticator.login(f"concurrent_user_{user_id}", "password123")

                        # Brief operation
                        await asyncio.sleep(0.001)  # Simulate work

                        # Logout
                        await authenticator.logout(token_str)

                        successes += 1

                    except Exception as e:
                        failures += 1
                        print(f"Concurrent operation failed for user {user_id}, op {op}: {e}")

                return {
                    "user_id": user_id,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": successes / (successes + failures) if (successes + failures) > 0 else 0,
                }

            # Run concurrent operations
            tasks = [user_operations(i) for i in range(concurrent_users)]
            round_results = await asyncio.gather(*tasks)

            # Analyze round results
            round_success_rates = [result["success_rate"] for result in round_results]
            round_avg_success = statistics.mean(round_success_rates)
            round_min_success = min(round_success_rates)

            stability_results.append(
                {
                    "round": round_num,
                    "avg_success_rate": round_avg_success,
                    "min_success_rate": round_min_success,
                    "user_results": round_results,
                }
            )

        # Analyze overall stability
        overall_avg_success = statistics.mean([r["avg_success_rate"] for r in stability_results])
        overall_min_success = min([r["min_success_rate"] for r in stability_results])
        success_rate_variance = statistics.variance([r["avg_success_rate"] for r in stability_results])

        assert overall_avg_success >= 0.95, f"Overall success rate {overall_avg_success:.3f} below 95%"
        assert overall_min_success >= 0.90, f"Min success rate {overall_min_success:.3f} below 90%"
        assert success_rate_variance < 0.01, f"Success rate variance {success_rate_variance:.6f} too high"

        # Generate concurrent stability report
        self._generate_concurrent_stability_report(
            {
                "test_config": {
                    "concurrent_users": concurrent_users,
                    "operations_per_user": operations_per_user,
                    "test_rounds": test_rounds,
                },
                "results": {
                    "overall_avg_success": overall_avg_success,
                    "overall_min_success": overall_min_success,
                    "success_rate_variance": success_rate_variance,
                    "round_results": stability_results,
                },
            }
        )

        # Clean up
        for i in range(concurrent_users):
            await authenticator.delete_user(f"concurrent_user_{i}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_recovery_stability(self, authenticator):
        """Verify stability of error recovery mechanisms."""
        from core.models.auth.enum import Role
        from core.exceptions import AuthenticationException
        from unittest.mock import Mock

        # Create test user
        await authenticator.create_user("recovery_user", "password123", {Role.USER})

        # Test 1: Invalid token recovery
        mock_request = Mock()
        error_recovery_success = 0
        total_recovery_attempts = 20

        for i in range(total_recovery_attempts):
            try:
                # Use invalid token
                mock_request.get_header.return_value = f"Bearer invalid_token_{i}"

                # This should fail
                with pytest.raises(AuthenticationException):
                    await authenticator.authenticate(mock_request)

                # System should still be functional after error
                token_str, auth_token = await authenticator.login("recovery_user", "password123")
                mock_request.get_header.return_value = f"Bearer {token_str}"
                validated_token = await authenticator.authenticate(mock_request)

                assert validated_token.username == "recovery_user"
                await authenticator.logout(token_str)

                error_recovery_success += 1

            except Exception as e:
                print(f"Error recovery failed on attempt {i}: {e}")

        recovery_rate = error_recovery_success / total_recovery_attempts
        assert recovery_rate >= 0.95, f"Error recovery rate {recovery_rate:.3f} below 95%"

        # Test 2: Token expiry recovery
        with time_machine.travel("2024-01-01 12:00:00", tick=True) as traveller:
            original_ttl = authenticator.default_token_ttl
            authenticator.default_token_ttl = 1  # 1 second

            try:
                expiry_recovery_success = 0
                expiry_attempts = 10

                for i in range(expiry_attempts):
                    try:
                        # Create short-lived token
                        token_str, auth_token = await authenticator.login("recovery_user", "password123")

                        # Wait for expiry using time machine
                        traveller.shift(1.5)

                        # Try to use expired token (should fail)
                        mock_request.get_header.return_value = f"Bearer {token_str}"
                        with pytest.raises(AuthenticationException):
                            await authenticator.authenticate(mock_request)

                        # System should recover and allow new login
                        new_token_str, new_auth_token = await authenticator.login("recovery_user", "password123")
                        assert new_auth_token.username == "recovery_user"
                        await authenticator.logout(new_token_str)

                        expiry_recovery_success += 1

                    except Exception as e:
                        print(f"Expiry recovery failed on attempt {i}: {e}")

                expiry_recovery_rate = expiry_recovery_success / expiry_attempts
                assert expiry_recovery_rate >= 0.90, f"Expiry recovery rate {expiry_recovery_rate:.3f} below 90%"

            finally:
                authenticator.default_token_ttl = original_ttl

        # Generate error recovery report
        self._generate_error_recovery_report(
            {
                "invalid_token_recovery": {"success_rate": recovery_rate, "total_attempts": total_recovery_attempts},
                "expiry_recovery": {"success_rate": expiry_recovery_rate, "total_attempts": expiry_attempts},
            }
        )

        # Clean up
        await authenticator.delete_user("recovery_user")

    def _generate_stability_report(self, metrics: Dict[str, Any]):
        """Generate stability verification report."""

        report_content = []
        report_content.append("# Authentication Stability Verification Report")
        report_content.append("")

        for operation, data in metrics.items():
            report_content.append(f"## {operation.title()} Stability")
            report_content.append(f"- Average Success Rate: {data['avg_success_rate']:.3f}")
            report_content.append(f"- Minimum Success Rate: {data['min_success_rate']:.3f}")
            if "time_variance" in data:
                report_content.append(f"- Time Variance: {data['time_variance']:.6f}")
            report_content.append(f"- Total Operations: {data['total_operations']}")
            report_content.append("")

        # Stability thresholds
        report_content.append("## Stability Thresholds")
        report_content.append("- Login: ≥99% average, ≥95% minimum")
        report_content.append("- Authentication: ≥99% average, ≥98% minimum")
        report_content.append("- Authorization: ≥99% average, ≥98% minimum")
        report_content.append("")

        # Save report
        report_file = Path(__file__).parent / "stability_verification_report.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_content))

    def _generate_concurrent_stability_report(self, data: Dict[str, Any]):
        """Generate concurrent stability report."""

        report_content = []
        report_content.append("# Concurrent Operation Stability Report")
        report_content.append("")

        # Test configuration
        config = data["test_config"]
        report_content.append("## Test Configuration")
        report_content.append(f"- Concurrent Users: {config['concurrent_users']}")
        report_content.append(f"- Operations per User: {config['operations_per_user']}")
        report_content.append(f"- Test Rounds: {config['test_rounds']}")
        report_content.append("")

        # Results
        results = data["results"]
        report_content.append("## Overall Results")
        report_content.append(f"- Overall Average Success: {results['overall_avg_success']:.3f}")
        report_content.append(f"- Overall Minimum Success: {results['overall_min_success']:.3f}")
        report_content.append(f"- Success Rate Variance: {results['success_rate_variance']:.6f}")
        report_content.append("")

        # Save report
        report_file = Path(__file__).parent / "concurrent_stability_report.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_content))

    def _generate_error_recovery_report(self, metrics: Dict[str, Any]):
        """Generate error recovery stability report."""

        report_content = []
        report_content.append("# Error Recovery Stability Report")
        report_content.append("")

        for recovery_type, data in metrics.items():
            report_content.append(f"## {recovery_type.replace('_', ' ').title()}")
            report_content.append(f"- Success Rate: {data['success_rate']:.3f}")
            report_content.append(f"- Total Attempts: {data['total_attempts']}")
            report_content.append("")

        # Save report
        report_file = Path(__file__).parent / "error_recovery_report.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_content))
