# ABOUTME: Benchmark tests for BaseCoreSettings validators using pytest-benchmark
# ABOUTME: Provides accurate performance measurements for validation logic

import pytest
import sys
import os

# Add the core package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config._base import BaseCoreSettings


class TestValidatorsBenchmark:
    """Benchmark test suite for BaseCoreSettings validators."""

    def test_env_validator_benchmark(self, benchmark):
        """Benchmark ENV validator performance."""
        def create_env_settings():
            return BaseCoreSettings(ENV="dev")
        
        result = benchmark(create_env_settings)
        assert result.ENV == "development"

    def test_env_validator_various_inputs_benchmark(self, benchmark):
        """Benchmark ENV validator with various inputs."""
        inputs = ["development", "dev", "PROD", "staging", "PRODUCTION"]
        
        def create_multiple_env_settings():
            results = []
            for env in inputs:
                results.append(BaseCoreSettings(ENV=env))
            return results
        
        results = benchmark(create_multiple_env_settings)
        assert len(results) == 5
        assert results[0].ENV == "development"
        assert results[1].ENV == "development"  # dev -> development
        assert results[2].ENV == "production"   # PROD -> production

    def test_log_level_validator_benchmark(self, benchmark):
        """Benchmark LOG_LEVEL validator performance."""
        def create_log_level_settings():
            return BaseCoreSettings(LOG_LEVEL="debug")
        
        result = benchmark(create_log_level_settings)
        assert result.LOG_LEVEL == "DEBUG"

    def test_log_format_validator_benchmark(self, benchmark):
        """Benchmark LOG_FORMAT validator performance."""
        def create_log_format_settings():
            return BaseCoreSettings(LOG_FORMAT="structured")
        
        result = benchmark(create_log_format_settings)
        assert result.LOG_FORMAT == "json"

    def test_timezone_validator_benchmark(self, benchmark):
        """Benchmark TIMEZONE validator performance."""
        def create_timezone_settings():
            return BaseCoreSettings(TIMEZONE="UTC")
        
        result = benchmark(create_timezone_settings)
        assert result.TIMEZONE == "UTC"

    def test_timezone_case_normalization_benchmark(self, benchmark):
        """Benchmark TIMEZONE validator with case normalization."""
        def create_timezone_case_settings():
            return BaseCoreSettings(TIMEZONE="america/new_york")
        
        result = benchmark(create_timezone_case_settings)
        assert result.TIMEZONE == "America/New_York"

    def test_full_settings_creation_benchmark(self, benchmark):
        """Benchmark complete settings object creation."""
        def create_full_settings():
            return BaseCoreSettings(
                APP_NAME="BenchmarkApp",
                ENV="prod",
                DEBUG=False,
                LOG_LEVEL="info",
                LOG_FORMAT="json",
                TIMEZONE="Asia/Shanghai"
            )
        
        result = benchmark(create_full_settings)
        assert result.APP_NAME == "BenchmarkApp"
        assert result.ENV == "production"
        assert result.LOG_LEVEL == "INFO"
        assert result.LOG_FORMAT == "json"
        assert result.TIMEZONE == "Asia/Shanghai"

    def test_default_settings_benchmark(self, benchmark):
        """Benchmark default settings creation (no validation needed)."""
        def create_default_settings():
            return BaseCoreSettings()
        
        result = benchmark(create_default_settings)
        assert result.APP_NAME == "TradingChart"
        assert result.ENV == "development"
        assert result.DEBUG is False
        assert result.LOG_LEVEL == "INFO"
        assert result.LOG_FORMAT == "txt"
        assert result.TIMEZONE == "UTC"

    @pytest.mark.parametrize("env_value", [
        "development", "dev", "develop", 
        "production", "prod", 
        "staging", "stage"
    ])
    def test_env_validator_parametrized_benchmark(self, benchmark, env_value):
        """Benchmark ENV validator with parametrized inputs."""
        def create_env_settings():
            return BaseCoreSettings(ENV=env_value)
        
        result = benchmark(create_env_settings)
        # Verify the mapping works correctly
        expected_mapping = {
            "development": "development",
            "dev": "development", 
            "develop": "development",
            "production": "production",
            "prod": "production",
            "staging": "staging",
            "stage": "staging"
        }
        assert result.ENV == expected_mapping[env_value.lower()]

    @pytest.mark.parametrize("timezone", [
        "UTC",
        "America/New_York",
        "Europe/London", 
        "Asia/Tokyo",
        "Australia/Sydney"
    ])
    def test_timezone_validator_parametrized_benchmark(self, benchmark, timezone):
        """Benchmark TIMEZONE validator with various timezones."""
        def create_timezone_settings():
            return BaseCoreSettings(TIMEZONE=timezone)
        
        result = benchmark(create_timezone_settings)
        assert result.TIMEZONE == timezone