# ABOUTME: Integration tests for configuration system
# ABOUTME: Tests caching, environment interactions, and configuration lifecycle

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config.settings import CoreSettings, get_settings


class TestSettingsCachingIntegration:
    """Integration tests for settings caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        get_settings.cache_clear()

    @pytest.mark.integration
    @pytest.mark.config
    def test_singleton_behavior_across_imports(self):
        """Test that settings behave as singleton across different import contexts."""
        # Clear cache first
        get_settings.cache_clear()
        
        # Simulate different import scenarios
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Import the global instance
        from core.config.settings import settings as global_settings
        
        # Function calls should return the same instance
        assert settings1 is settings2
        
        # Global settings should have same configuration values
        assert global_settings.APP_NAME == settings1.APP_NAME
        assert global_settings.ENV == settings1.ENV
        assert global_settings.DEBUG == settings1.DEBUG

    @pytest.mark.integration
    @pytest.mark.config
    def test_cache_persistence_across_environment_changes(self):
        """Test how caching behaves when environment changes."""
        # Clear cache
        get_settings.cache_clear()
        
        # Get settings with default environment
        settings1 = get_settings()
        original_app_name = settings1.APP_NAME
        
        # Change environment variable
        with patch.dict(os.environ, {"APP_NAME": "ChangedApp"}):
            # Cache should still return the same instance
            settings2 = get_settings()
            assert settings1 is settings2
            assert settings2.APP_NAME == original_app_name  # Still cached value
            
            # Clear cache and get new settings
            get_settings.cache_clear()
            settings3 = get_settings()
            assert settings3.APP_NAME == "ChangedApp"  # New value from env

    @pytest.mark.integration
    @pytest.mark.config
    def test_cache_behavior_with_direct_instantiation(self):
        """Test cache behavior vs direct CoreSettings instantiation."""
        # Clear cache
        get_settings.cache_clear()
        
        # Get cached instance
        cached_settings = get_settings()
        
        # Create direct instance
        direct_settings = CoreSettings()
        
        # Should be different instances
        assert cached_settings is not direct_settings
        
        # But should have same values (assuming no env changes)
        assert cached_settings.APP_NAME == direct_settings.APP_NAME
        assert cached_settings.ENV == direct_settings.ENV

    @pytest.mark.integration
    @pytest.mark.config
    def test_cache_thread_safety(self):
        """Test that caching is thread-safe."""
        # Clear cache
        get_settings.cache_clear()
        
        results = []
        
        def get_settings_in_thread():
            """Function to call get_settings() in a thread."""
            return get_settings()
        
        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_settings_in_thread) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        # All results should have the same configuration values
        first_result = results[0]
        for result in results[1:]:
            assert result.APP_NAME == first_result.APP_NAME
            assert result.ENV == first_result.ENV
            assert result.DEBUG == first_result.DEBUG
            assert isinstance(result, CoreSettings)

    @pytest.mark.integration
    @pytest.mark.config
    def test_cache_memory_efficiency(self):
        """Test that cache doesn't create multiple instances."""
        # Clear cache
        get_settings.cache_clear()
        
        # Get multiple references
        settings_list = [get_settings() for _ in range(100)]
        
        # All should be the same instance
        first_settings = settings_list[0]
        for settings in settings_list[1:]:
            assert settings is first_settings
        
        # Cache should only have one item
        cache_info = get_settings.cache_info()
        assert cache_info.currsize == 1

    @pytest.mark.integration
    @pytest.mark.config
    def test_cache_clear_and_repopulate(self):
        """Test cache clearing and repopulation cycle."""
        # Initial population
        settings1 = get_settings()
        cache_info1 = get_settings.cache_info()
        assert cache_info1.currsize == 1
        
        # Clear cache
        get_settings.cache_clear()
        cache_info2 = get_settings.cache_info()
        assert cache_info2.currsize == 0
        
        # Repopulate
        settings2 = get_settings()
        cache_info3 = get_settings.cache_info()
        assert cache_info3.currsize == 1
        
        # Should be different instances
        assert settings1 is not settings2


class TestEnvironmentIntegration:
    """Integration tests for environment variable and .env file interactions."""

    def setup_method(self):
        """Clear cache before each test."""
        get_settings.cache_clear()

    @pytest.mark.integration
    @pytest.mark.config
    def test_environment_variable_priority_integration(self):
        """Test environment variable priority in realistic scenarios."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create .env file
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("""
APP_NAME=DotEnvApp
ENV=development
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=txt
TIMEZONE=UTC
""")
            
            # Set some environment variables
            with patch.dict(os.environ, {
                "ENV": "production",
                "DEBUG": "true",
                "LOG_LEVEL": "ERROR"
            }):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmp_dir)
                    settings = get_settings()
                    
                    # Environment variables should override .env file
                    assert settings.APP_NAME == "DotEnvApp"  # From .env
                    assert settings.ENV == "production"      # From env var
                    assert settings.DEBUG is True           # From env var
                    assert settings.LOG_LEVEL == "ERROR"    # From env var
                    assert settings.LOG_FORMAT == "txt"     # From .env
                    assert settings.TIMEZONE == "UTC"       # From .env
                finally:
                    os.chdir(original_cwd)

    @pytest.mark.integration
    @pytest.mark.config
    def test_case_insensitive_environment_integration(self):
        """Test case-insensitive environment variable handling integration."""
        with patch.dict(os.environ, {
            "app_name": "LowerCaseApp",
            "ENV": "PROD",
            "debug": "TRUE",
            "log_level": "warning",
            "LOG_FORMAT": "STRUCTURED"
        }):
            settings = get_settings()
            
            # Should handle case-insensitive env vars and values
            assert settings.APP_NAME == "LowerCaseApp"
            assert settings.ENV == "production"
            assert settings.DEBUG is True
            assert settings.LOG_LEVEL == "WARNING"
            assert settings.LOG_FORMAT == "json"

    @pytest.mark.integration
    @pytest.mark.config
    def test_malformed_environment_handling_integration(self):
        """Test handling of malformed environment variables."""
        with patch.dict(os.environ, {
            "APP_NAME": "ValidApp",
            "ENV": "invalid_environment",  # Invalid
            "DEBUG": "not_a_boolean",      # Invalid
            "LOG_LEVEL": "INVALID_LEVEL",  # Invalid
            "LOG_FORMAT": "invalid_format" # Invalid
        }):
            # Should raise ValidationError for invalid values
            with pytest.raises(Exception):  # Pydantic ValidationError
                get_settings()

    @pytest.mark.integration
    @pytest.mark.config
    def test_unicode_environment_integration(self):
        """Test Unicode support in environment variables."""
        with patch.dict(os.environ, {
            "APP_NAME": "交易系統",
            "TIMEZONE": "Asia/Shanghai"
        }):
            settings = get_settings()
            assert settings.APP_NAME == "交易系統"
            assert settings.TIMEZONE == "Asia/Shanghai"

    @pytest.mark.integration
    @pytest.mark.config
    def test_environment_change_detection(self):
        """Test that environment changes require cache clear to take effect."""
        # Initial settings
        with patch.dict(os.environ, {"APP_NAME": "InitialApp"}):
            settings1 = get_settings()
            assert settings1.APP_NAME == "InitialApp"
        
        # Change environment without clearing cache
        with patch.dict(os.environ, {"APP_NAME": "ChangedApp"}):
            settings2 = get_settings()
            assert settings2.APP_NAME == "InitialApp"  # Still cached
            assert settings1 is settings2
        
        # Clear cache and get new settings
        get_settings.cache_clear()
        with patch.dict(os.environ, {"APP_NAME": "ChangedApp"}):
            settings3 = get_settings()
            assert settings3.APP_NAME == "ChangedApp"  # New value


class TestConfigurationLifecycle:
    """Integration tests for complete configuration lifecycle."""

    def setup_method(self):
        """Clear cache before each test."""
        get_settings.cache_clear()

    @pytest.mark.integration
    @pytest.mark.config
    def test_application_startup_simulation(self):
        """Simulate application startup configuration loading."""
        # Simulate application startup with various configuration sources
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create .env file (like in development)
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("""
APP_NAME=DevApp
ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=txt
""")
            
            # Set production environment variables (like in deployment)
            with patch.dict(os.environ, {
                "ENV": "production",
                "DEBUG": "false",
                "LOG_LEVEL": "INFO",
                "LOG_FORMAT": "json"
            }):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmp_dir)
                    
                    # First access - should load configuration
                    settings = get_settings()
                    assert settings.APP_NAME == "DevApp"        # From .env
                    assert settings.ENV == "production"        # From env var
                    assert settings.DEBUG is False             # From env var
                    assert settings.LOG_LEVEL == "INFO"        # From env var
                    assert settings.LOG_FORMAT == "json"       # From env var
                    
                    # Subsequent accesses should be cached
                    settings2 = get_settings()
                    assert settings is settings2
                    
                    # Cache info should show hits
                    cache_info = get_settings.cache_info()
                    assert cache_info.hits > 0
                    assert cache_info.currsize == 1
                    
                finally:
                    os.chdir(original_cwd)

    @pytest.mark.integration
    @pytest.mark.config
    def test_configuration_validation_pipeline(self):
        """Test the complete validation pipeline."""
        # Test with various input formats that should all be normalized
        with patch.dict(os.environ, {
            "ENV": "  PROD  ",                    # Whitespace + case
            "LOG_LEVEL": "warning",               # Lowercase
            "LOG_FORMAT": "STRUCTURED",           # Uppercase alias
            "DEBUG": "1"                          # String boolean
        }):
            settings = get_settings()
            
            # All should be properly normalized
            assert settings.ENV == "production"
            assert settings.LOG_LEVEL == "WARNING"
            assert settings.LOG_FORMAT == "json"
            assert settings.DEBUG is True

    @pytest.mark.integration
    @pytest.mark.config
    def test_configuration_extension_integration(self):
        """Test configuration extension patterns work with caching."""
        # Test that extended configuration classes work with the caching system
        from pydantic import Field
        
        class ExtendedCoreSettings(CoreSettings):
            DATABASE_URL: str = Field(default="sqlite:///test.db")
            REDIS_URL: str = Field(default="redis://localhost:6379")
        
        # Clear the original cache
        get_settings.cache_clear()
        
        # Extended settings should work independently
        extended_settings = ExtendedCoreSettings()
        assert extended_settings.DATABASE_URL == "sqlite:///test.db"
        assert extended_settings.REDIS_URL == "redis://localhost:6379"
        
        # Should still have base functionality
        assert extended_settings.APP_NAME == "TradingChart"
        assert extended_settings.ENV == "development"

    @pytest.mark.integration
    @pytest.mark.config
    def test_configuration_serialization_integration(self):
        """Test configuration serialization in realistic scenarios."""
        with patch.dict(os.environ, {
            "APP_NAME": "SerializationTest",
            "ENV": "production",
            "DEBUG": "false"
        }):
            settings = get_settings()
            
            # Test dictionary serialization
            config_dict = settings.model_dump()
            assert isinstance(config_dict, dict)
            assert config_dict["APP_NAME"] == "SerializationTest"
            assert config_dict["ENV"] == "production"
            assert config_dict["DEBUG"] is False
            
            # Test JSON serialization
            config_json = settings.model_dump_json()
            assert isinstance(config_json, str)
            assert "SerializationTest" in config_json
            assert "production" in config_json
            
            # Test that serialization is consistent across cache hits
            settings2 = get_settings()  # Should be cached
            assert settings is settings2
            assert settings2.model_dump() == config_dict

    @pytest.mark.integration
    @pytest.mark.config
    def test_configuration_error_handling_integration(self):
        """Test error handling in realistic configuration scenarios."""
        # Test partial invalid configuration
        with patch.dict(os.environ, {
            "APP_NAME": "ValidApp",
            "ENV": "production",        # Valid
            "DEBUG": "false",           # Valid
            "LOG_LEVEL": "INVALID",     # Invalid
            "LOG_FORMAT": "json"        # Valid
        }):
            with pytest.raises(Exception):  # Should raise ValidationError
                get_settings()
        
        # Cache should not be populated after error
        cache_info = get_settings.cache_info()
        assert cache_info.currsize == 0

    @pytest.mark.integration
    @pytest.mark.config
    def test_concurrent_configuration_access(self):
        """Test concurrent access to configuration."""
        # Clear cache
        get_settings.cache_clear()
        
        # Set up environment
        with patch.dict(os.environ, {
            "APP_NAME": "ConcurrentApp",
            "ENV": "production"
        }):
            results = []
            
            def access_settings():
                """Access settings from multiple threads."""
                time.sleep(0.01)  # Small delay to increase chance of race conditions
                return get_settings()
            
            # Run concurrent accesses
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(access_settings) for _ in range(10)]
                results = [future.result() for future in as_completed(futures)]
            
            # All should have correct values (identity check may fail due to concurrency)
            for result in results:
                assert result.APP_NAME == "ConcurrentApp"
                assert result.ENV == "production"
                assert isinstance(result, CoreSettings)
            
            # Verify cache efficiency - should have minimal cache misses
            cache_info = get_settings.cache_info()
            assert cache_info.currsize == 1  # Should only have one cached instance
