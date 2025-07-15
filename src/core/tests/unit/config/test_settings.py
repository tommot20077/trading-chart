# ABOUTME: Unit tests for CoreSettings class and configuration composition
# ABOUTME: Tests inheritance, instantiation, and basic functionality of CoreSettings

import pytest
from unittest.mock import patch
from core.config.settings import CoreSettings, get_settings
from core.config._base import BaseCoreSettings


class TestCoreSettings:
    """Test suite for CoreSettings class."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_inheritance(self):
        """Test that CoreSettings properly inherits from BaseCoreSettings."""
        settings = CoreSettings()

        # Should inherit all fields from BaseCoreSettings
        assert hasattr(settings, "APP_NAME")
        assert hasattr(settings, "ENV")
        assert hasattr(settings, "DEBUG")
        assert hasattr(settings, "LOG_LEVEL")
        assert hasattr(settings, "LOG_FORMAT")
        assert hasattr(settings, "TIMEZONE")

        # Should have the same default values
        assert settings.APP_NAME == "TradingChart"
        assert settings.ENV == "development"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "INFO"
        assert settings.LOG_FORMAT == "txt"
        assert settings.TIMEZONE == "UTC"

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_is_instance_of_basecoresettings(self):
        """Test that CoreSettings is an instance of BaseCoreSettings."""
        settings = CoreSettings()
        assert isinstance(settings, BaseCoreSettings)
        assert isinstance(settings, CoreSettings)

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_with_custom_values(self):
        """Test CoreSettings with custom initialization values."""
        settings = CoreSettings(
            APP_NAME="CustomApp",
            ENV="production",
            DEBUG=True,
            LOG_LEVEL="ERROR",
            LOG_FORMAT="json",
            TIMEZONE="Asia/Tokyo",
        )

        assert settings.APP_NAME == "CustomApp"
        assert settings.ENV == "production"
        assert settings.DEBUG is True
        assert settings.LOG_LEVEL == "ERROR"
        assert settings.LOG_FORMAT == "json"
        assert settings.TIMEZONE == "Asia/Tokyo"

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_validation_inheritance(self):
        """Test that CoreSettings inherits all validation from BaseCoreSettings."""
        # Test case-insensitive ENV validation
        settings = CoreSettings(ENV="prod")
        assert settings.ENV == "production"

        # Test case-insensitive LOG_LEVEL validation
        settings = CoreSettings(LOG_LEVEL="debug")
        assert settings.LOG_LEVEL == "DEBUG"

        # Test case-insensitive LOG_FORMAT validation
        settings = CoreSettings(LOG_FORMAT="structured")
        assert settings.LOG_FORMAT == "json"

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_invalid_values(self):
        """Test that CoreSettings validation raises errors for invalid values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CoreSettings(ENV="invalid_env")

        with pytest.raises(ValidationError):
            CoreSettings(LOG_LEVEL="invalid_level")

        with pytest.raises(ValidationError):
            CoreSettings(LOG_FORMAT="invalid_format")

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_environment_variable_loading(self):
        """Test that CoreSettings loads from environment variables."""
        import os

        with patch.dict(
            os.environ,
            {"APP_NAME": "EnvApp", "ENV": "staging", "DEBUG": "true", "LOG_LEVEL": "warning", "LOG_FORMAT": "json"},
        ):
            settings = CoreSettings()
            assert settings.APP_NAME == "EnvApp"
            assert settings.ENV == "staging"
            assert settings.DEBUG is True
            assert settings.LOG_LEVEL == "WARNING"
            assert settings.LOG_FORMAT == "json"

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_model_config_inheritance(self):
        """Test that CoreSettings inherits model configuration from BaseCoreSettings."""
        settings = CoreSettings()

        # Check that model_config is properly inherited
        assert hasattr(settings, "model_config")
        config = settings.model_config

        # Should inherit configuration from BaseCoreSettings
        assert config.get("env_file") == ".env"
        assert config.get("env_file_encoding") == "utf-8"
        assert config.get("case_sensitive") is False
        assert config.get("extra") == "ignore"

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_serialization(self):
        """Test CoreSettings serialization to dict and JSON."""
        settings = CoreSettings(APP_NAME="TestApp", ENV="production", DEBUG=True)

        # Test dict serialization
        settings_dict = settings.model_dump()
        assert isinstance(settings_dict, dict)
        assert settings_dict["APP_NAME"] == "TestApp"
        assert settings_dict["ENV"] == "production"
        assert settings_dict["DEBUG"] is True

        # Test JSON serialization
        json_str = settings.model_dump_json()
        assert isinstance(json_str, str)
        assert "TestApp" in json_str
        assert "production" in json_str

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_field_access(self):
        """Test accessing CoreSettings fields and metadata."""
        settings = CoreSettings()

        # Test field access
        assert settings.APP_NAME == "TradingChart"
        assert settings.ENV == "development"

        # Test field metadata access
        fields = CoreSettings.model_fields
        assert "APP_NAME" in fields
        assert "ENV" in fields
        assert fields["APP_NAME"].description is not None

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_extension_pattern(self):
        """Test that CoreSettings can be extended following the documented pattern."""
        # This tests the pattern shown in the docstring
        from pydantic import Field

        class ExtendedSettings(CoreSettings):
            CUSTOM_FIELD: str = Field(default="custom_value", description="Custom field for testing")

        settings = ExtendedSettings()

        # Should have all base fields
        assert settings.APP_NAME == "TradingChart"
        assert settings.ENV == "development"

        # Should have custom field
        assert settings.CUSTOM_FIELD == "custom_value"

    @pytest.mark.unit
    @pytest.mark.config
    def test_coresettings_immutability_behavior(self):
        """Test CoreSettings field assignment behavior."""
        settings = CoreSettings()
        original_app_name = settings.APP_NAME

        # Test field assignment (should work in Pydantic v2 by default)
        settings.APP_NAME = "Modified"
        assert settings.APP_NAME == "Modified"

        # Reset for other tests
        settings.APP_NAME = original_app_name


class TestGetSettingsFunction:
    """Test suite for get_settings() function."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_returns_coresettings_instance(self):
        """Test that get_settings() returns a CoreSettings instance."""
        settings = get_settings()
        assert isinstance(settings, CoreSettings)
        assert isinstance(settings, BaseCoreSettings)

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_with_clean_cache(self):
        """Test get_settings() with clean cache state."""
        # Clear the cache first
        get_settings.cache_clear()

        settings = get_settings()
        assert isinstance(settings, CoreSettings)
        assert settings.APP_NAME == "TradingChart"

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_function_signature(self):
        """Test get_settings() function signature and return type annotation."""
        import inspect

        sig = inspect.signature(get_settings)
        assert len(sig.parameters) == 0  # No parameters

        # Check return type annotation
        assert sig.return_annotation == CoreSettings

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_with_environment_variables(self):
        """Test get_settings() with environment variables."""
        import os

        # Clear cache before test
        get_settings.cache_clear()

        with patch.dict(os.environ, {"APP_NAME": "FunctionTestApp", "ENV": "production"}):
            settings = get_settings()
            assert settings.APP_NAME == "FunctionTestApp"
            assert settings.ENV == "production"

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_cache_behavior_basic(self):
        """Test basic caching behavior of get_settings()."""
        # Clear cache
        get_settings.cache_clear()

        # Get settings twice
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance due to caching
        assert settings1 is settings2

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_cache_info(self):
        """Test cache info functionality."""
        # Clear cache
        get_settings.cache_clear()

        # Check initial cache info
        cache_info = get_settings.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 0
        assert cache_info.currsize == 0

        # Call function once
        get_settings()
        cache_info = get_settings.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 1
        assert cache_info.currsize == 1

        # Call function again
        get_settings()
        cache_info = get_settings.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 1
        assert cache_info.currsize == 1

    @pytest.mark.unit
    @pytest.mark.config
    def test_get_settings_cache_clear(self):
        """Test cache clearing functionality."""
        # Get settings to populate cache
        get_settings()

        # Verify cache is populated
        cache_info = get_settings.cache_info()
        assert cache_info.currsize == 1

        # Clear cache
        get_settings.cache_clear()

        # Verify cache is cleared
        cache_info = get_settings.cache_info()
        assert cache_info.currsize == 0


class TestGlobalSettingsInstance:
    """Test suite for the global settings instance."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_global_settings_instance_exists(self):
        """Test that global settings instance is properly defined."""
        from core.config.settings import settings

        assert settings is not None
        assert isinstance(settings, CoreSettings)
        assert isinstance(settings, BaseCoreSettings)

    @pytest.mark.unit
    @pytest.mark.config
    def test_global_settings_instance_values(self):
        """Test that global settings instance has correct values."""
        from core.config.settings import settings

        # Should have default values (unless overridden by environment)
        assert isinstance(settings.APP_NAME, str)
        assert settings.ENV in ["development", "staging", "production"]
        assert isinstance(settings.DEBUG, bool)
        assert settings.LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert settings.LOG_FORMAT in ["json", "txt"]
        assert isinstance(settings.TIMEZONE, str)

    @pytest.mark.unit
    @pytest.mark.config
    def test_global_settings_is_from_get_settings(self):
        """Test that global settings instance comes from get_settings()."""
        from core.config.settings import settings

        # Global settings should be a CoreSettings instance
        assert isinstance(settings, CoreSettings)

        # Should have the same configuration as get_settings() returns
        function_settings = get_settings()
        assert settings.APP_NAME == function_settings.APP_NAME
        assert settings.ENV == function_settings.ENV
        assert settings.DEBUG == function_settings.DEBUG
        assert settings.LOG_LEVEL == function_settings.LOG_LEVEL
        assert settings.LOG_FORMAT == function_settings.LOG_FORMAT
        assert settings.TIMEZONE == function_settings.TIMEZONE

    @pytest.mark.unit
    @pytest.mark.config
    def test_global_settings_import_accessibility(self):
        """Test that global settings can be imported and used."""
        # This simulates how other modules would import settings
        from core.config.settings import settings

        # Should be usable immediately
        app_name = settings.APP_NAME
        assert isinstance(app_name, str)

        # Should be able to access all fields
        assert hasattr(settings, "ENV")
        assert hasattr(settings, "DEBUG")
        assert hasattr(settings, "LOG_LEVEL")
        assert hasattr(settings, "LOG_FORMAT")
        assert hasattr(settings, "TIMEZONE")
