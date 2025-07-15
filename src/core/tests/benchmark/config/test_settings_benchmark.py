# ABOUTME: Benchmark tests for CoreSettings and singleton performance
# ABOUTME: Tests the performance of settings composition and caching

import pytest
import sys
import os

# Add the core package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.config.settings import CoreSettings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Clear the settings cache before each test to ensure clean state."""
    # Clear the lru_cache to ensure fresh settings for each test
    get_settings.cache_clear()
    yield
    # Clean up after test
    get_settings.cache_clear()


class TestSettingsBenchmark:
    """Benchmark test suite for CoreSettings composition and caching."""

    @pytest.mark.unit
    @pytest.mark.benchmark
    @pytest.mark.config
    def test_core_settings_creation_benchmark(self, benchmark):
        """Benchmark CoreSettings creation (inherits from BaseCoreSettings)."""

        def create_core_settings():
            return CoreSettings()

        result = benchmark(create_core_settings)
        assert result.APP_NAME == "TradingChart"
        assert result.ENV == "development"

    @pytest.mark.unit
    @pytest.mark.benchmark
    @pytest.mark.config
    def test_core_settings_with_overrides_benchmark(self, benchmark):
        """Benchmark CoreSettings creation with field overrides."""

        def create_core_settings_with_overrides():
            return CoreSettings(ENV="production", LOG_LEVEL="ERROR", TIMEZONE="Europe/London")

        result = benchmark(create_core_settings_with_overrides)
        assert result.ENV == "production"
        assert result.LOG_LEVEL == "ERROR"
        assert result.TIMEZONE == "Europe/London"

    @pytest.mark.unit
    @pytest.mark.benchmark
    @pytest.mark.config
    def test_get_settings_singleton_benchmark(self, benchmark):
        """Benchmark get_settings() singleton function."""

        def get_singleton_settings():
            return get_settings()

        result = benchmark(get_singleton_settings)
        assert isinstance(result, CoreSettings)
        assert result.APP_NAME == "TradingChart"

    @pytest.mark.unit
    @pytest.mark.benchmark
    @pytest.mark.config
    def test_get_settings_multiple_calls_benchmark(self, benchmark):
        """Benchmark multiple calls to get_settings() (should be cached)."""

        def get_multiple_settings():
            results = []
            for _ in range(10):
                results.append(get_settings())
            return results

        results = benchmark(get_multiple_settings)
        assert len(results) == 10
        # All should be the same instance due to caching
        first_instance = results[0]
        for result in results[1:]:
            assert result is first_instance

    @pytest.mark.unit
    @pytest.mark.benchmark
    @pytest.mark.config
    def test_settings_attribute_access_benchmark(self, benchmark):
        """Benchmark attribute access on settings object."""
        settings = get_settings()

        def access_settings_attributes():
            return (
                settings.APP_NAME,
                settings.ENV,
                settings.DEBUG,
                settings.LOG_LEVEL,
                settings.LOG_FORMAT,
                settings.TIMEZONE,
            )

        result = benchmark(access_settings_attributes)
        assert result[0] == "TradingChart"  # APP_NAME
        assert result[1] == "development"  # ENV
        assert result[2] is False  # DEBUG
        assert result[3] == "INFO"  # LOG_LEVEL
        assert result[4] == "txt"  # LOG_FORMAT
        assert result[5] == "UTC"  # TIMEZONE
