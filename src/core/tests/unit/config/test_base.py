# ABOUTME: Unit tests for base configuration settings
# ABOUTME: Tests case-insensitive validation and environment variable handling
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from core.config._base import BaseCoreSettings


class TestBaseCoreSettings:
    """Test suite for BaseCoreSettings configuration class."""

    @pytest.mark.unit
    @pytest.mark.config
    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = BaseCoreSettings()

        assert settings.APP_NAME == "TradingChart"
        assert settings.ENV == "development"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "INFO"
        assert settings.LOG_FORMAT == "txt"
        assert settings.TIMEZONE == "UTC"

    @pytest.mark.unit
    @pytest.mark.config
    def test_env_case_insensitive_validation(self):
        """Test ENV field accepts case-insensitive values and aliases."""
        # Test exact matches (case insensitive)
        settings = BaseCoreSettings(ENV="DEVELOPMENT")
        assert settings.ENV == "development"

        settings = BaseCoreSettings(ENV="Production")
        assert settings.ENV == "production"

        settings = BaseCoreSettings(ENV="STAGING")
        assert settings.ENV == "staging"

        # Test aliases
        settings = BaseCoreSettings(ENV="dev")
        assert settings.ENV == "development"

        settings = BaseCoreSettings(ENV="DEV")
        assert settings.ENV == "development"

        settings = BaseCoreSettings(ENV="develop")
        assert settings.ENV == "development"

        settings = BaseCoreSettings(ENV="prod")
        assert settings.ENV == "production"

        settings = BaseCoreSettings(ENV="PROD")
        assert settings.ENV == "production"

        settings = BaseCoreSettings(ENV="stage")
        assert settings.ENV == "staging"

        settings = BaseCoreSettings(ENV="STAGE")
        assert settings.ENV == "staging"

    @pytest.mark.unit
    @pytest.mark.config
    def test_env_validation_with_whitespace(self):
        """Test ENV field handles whitespace correctly."""
        settings = BaseCoreSettings(ENV="  dev  ")
        assert settings.ENV == "development"

        settings = BaseCoreSettings(ENV="\tproduction\n")
        assert settings.ENV == "production"

    @pytest.mark.unit
    @pytest.mark.config
    def test_env_invalid_value(self):
        """Test ENV field validation fails for invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCoreSettings(ENV="invalid_env")

        assert "Input should be 'development', 'staging' or 'production'" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_level_case_insensitive_validation(self):
        """Test LOG_LEVEL field accepts case-insensitive values."""
        settings = BaseCoreSettings(LOG_LEVEL="debug")
        assert settings.LOG_LEVEL == "DEBUG"

        settings = BaseCoreSettings(LOG_LEVEL="info")
        assert settings.LOG_LEVEL == "INFO"

        settings = BaseCoreSettings(LOG_LEVEL="warning")
        assert settings.LOG_LEVEL == "WARNING"

        settings = BaseCoreSettings(LOG_LEVEL="error")
        assert settings.LOG_LEVEL == "ERROR"

        settings = BaseCoreSettings(LOG_LEVEL="critical")
        assert settings.LOG_LEVEL == "CRITICAL"

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_level_validation_with_whitespace(self):
        """Test LOG_LEVEL field handles whitespace correctly."""
        settings = BaseCoreSettings(LOG_LEVEL="  debug  ")
        assert settings.LOG_LEVEL == "DEBUG"

        settings = BaseCoreSettings(LOG_LEVEL="\tinfo\n")
        assert settings.LOG_LEVEL == "INFO"

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_level_invalid_value(self):
        """Test LOG_LEVEL field validation fails for invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCoreSettings(LOG_LEVEL="INVALID")

        assert "Input should be 'DEBUG', 'INFO', 'WARNING', 'ERROR' or 'CRITICAL'" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_format_case_insensitive_validation(self):
        """Test LOG_FORMAT field accepts case-insensitive values and aliases."""
        # Test exact matches (case insensitive)
        settings = BaseCoreSettings(LOG_FORMAT="JSON")
        assert settings.LOG_FORMAT == "json"

        settings = BaseCoreSettings(LOG_FORMAT="Txt")
        assert settings.LOG_FORMAT == "txt"

        # Test aliases
        settings = BaseCoreSettings(LOG_FORMAT="text")
        assert settings.LOG_FORMAT == "txt"

        settings = BaseCoreSettings(LOG_FORMAT="TEXT")
        assert settings.LOG_FORMAT == "txt"

        settings = BaseCoreSettings(LOG_FORMAT="structured")
        assert settings.LOG_FORMAT == "json"

        settings = BaseCoreSettings(LOG_FORMAT="STRUCTURED")
        assert settings.LOG_FORMAT == "json"

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_format_validation_with_whitespace(self):
        """Test LOG_FORMAT field handles whitespace correctly."""
        settings = BaseCoreSettings(LOG_FORMAT="  json  ")
        assert settings.LOG_FORMAT == "json"

        settings = BaseCoreSettings(LOG_FORMAT="\ttxt\n")
        assert settings.LOG_FORMAT == "txt"

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_format_invalid_value(self):
        """Test LOG_FORMAT field validation fails for invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCoreSettings(LOG_FORMAT="invalid_format")

        assert "Input should be 'json' or 'txt'" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.config
    def test_combined_case_insensitive_validation(self):
        """Test multiple fields with case-insensitive values work together."""
        settings = BaseCoreSettings(ENV="PROD", LOG_LEVEL="warning", LOG_FORMAT="STRUCTURED")

        assert settings.ENV == "production"
        assert settings.LOG_LEVEL == "WARNING"
        assert settings.LOG_FORMAT == "json"

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_iana_names(self):
        """Test TIMEZONE field accepts valid IANA timezone names."""
        # Test common IANA timezone names
        valid_timezones = [
            "UTC",
            "America/New_York",
            "America/Los_Angeles",
            "Europe/London",
            "Europe/Paris",
            "Asia/Tokyo",
            "Asia/Shanghai",
            "Australia/Sydney",
            "Africa/Cairo",
        ]

        for tz in valid_timezones:
            settings = BaseCoreSettings(TIMEZONE=tz)
            assert settings.TIMEZONE == tz

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_case_normalization(self):
        """Test TIMEZONE field normalizes case for IANA names."""
        # Test case normalization
        test_cases = [
            ("america/new_york", "America/New_York"),
            ("europe/london", "Europe/London"),
            ("asia/shanghai", "Asia/Shanghai"),
            ("australia/sydney", "Australia/Sydney"),
        ]

        for input_tz, expected_tz in test_cases:
            settings = BaseCoreSettings(TIMEZONE=input_tz)
            assert settings.TIMEZONE == expected_tz

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_iana_only(self):
        """Test TIMEZONE field only accepts valid IANA timezone names."""
        # Test that only valid IANA names work
        valid_iana_names = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo", "Australia/Sydney"]

        for tz in valid_iana_names:
            settings = BaseCoreSettings(TIMEZONE=tz)
            assert settings.TIMEZONE == tz

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_with_whitespace(self):
        """Test TIMEZONE field handles whitespace correctly."""
        settings = BaseCoreSettings(TIMEZONE="  UTC  ")
        assert settings.TIMEZONE == "UTC"

        settings = BaseCoreSettings(TIMEZONE="\tAmerica/New_York\n")
        assert settings.TIMEZONE == "America/New_York"

        settings = BaseCoreSettings(TIMEZONE="  UTC  ")
        assert settings.TIMEZONE == "UTC"

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_invalid_values(self):
        """Test TIMEZONE field validation fails for invalid values."""
        invalid_timezones = [
            "Invalid/Timezone",
            "NotATimezone",
            "America/FakeCity",
            "Europe/NonExistent",
            "Random_String",
        ]

        for invalid_tz in invalid_timezones:
            with pytest.raises(ValidationError) as exc_info:
                BaseCoreSettings(TIMEZONE=invalid_tz)

            assert "Invalid timezone" in str(exc_info.value)

        # Test empty string separately (different error message)
        with pytest.raises(ValidationError) as exc_info:
            BaseCoreSettings(TIMEZONE="")

        # Empty string has different error from zoneinfo
        assert "normalized relative paths" in str(exc_info.value) or "Invalid timezone" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_edge_cases(self):
        """Test TIMEZONE field validation handles edge cases."""
        # Test with proper IANA names that should work
        edge_cases = [
            "America/Argentina/Buenos_Aires",  # Multi-level timezone
            "Pacific/Marquesas",  # Less common timezone
            "Indian/Maldives",  # Indian Ocean timezone
        ]

        for tz in edge_cases:
            settings = BaseCoreSettings(TIMEZONE=tz)
            assert settings.TIMEZONE == tz

    @pytest.mark.unit
    @pytest.mark.config
    def test_non_string_values_passthrough(self):
        """Test that non-string values are passed through without modification."""
        # This shouldn't happen in normal usage, but validators should handle it gracefully

        # Test that the validators handle non-string input gracefully
        # (though this would typically be caught by Pydantic's type validation first)
        validator = BaseCoreSettings.validate_env_case_insensitive
        assert validator(123) == 123  # Non-string should pass through

        validator = BaseCoreSettings.validate_log_level_case_insensitive
        assert validator(None) is None  # None should pass through

        validator = BaseCoreSettings.validate_log_format_case_insensitive
        assert validator(True) is True  # Boolean should pass through

        validator = BaseCoreSettings.validate_timezone
        assert validator(None) is None  # Non-string should pass through

    @pytest.mark.unit
    @pytest.mark.config
    def test_app_name_case_sensitivity(self):
        """Test APP_NAME field maintains case sensitivity."""
        settings = BaseCoreSettings(APP_NAME="TradingChart")
        assert settings.APP_NAME == "TradingChart"

        settings = BaseCoreSettings(APP_NAME="tradingchart")
        assert settings.APP_NAME == "tradingchart"

        settings = BaseCoreSettings(APP_NAME="TRADINGCHART")
        assert settings.APP_NAME == "TRADINGCHART"

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_case_sensitivity(self):
        """Test TIMEZONE field normalizes case for IANA names."""
        settings = BaseCoreSettings(TIMEZONE="UTC")
        assert settings.TIMEZONE == "UTC"

        settings = BaseCoreSettings(TIMEZONE="America/New_York")
        assert settings.TIMEZONE == "America/New_York"

        # Case normalization for IANA names
        settings = BaseCoreSettings(TIMEZONE="asia/shanghai")
        assert settings.TIMEZONE == "Asia/Shanghai"

    @pytest.mark.unit
    @pytest.mark.config
    def test_debug_boolean_validation(self):
        """Test DEBUG field accepts various boolean representations."""
        # Test string representations
        settings = BaseCoreSettings(DEBUG="true")
        assert settings.DEBUG is True

        settings = BaseCoreSettings(DEBUG="false")
        assert settings.DEBUG is False

        settings = BaseCoreSettings(DEBUG="True")
        assert settings.DEBUG is True

        settings = BaseCoreSettings(DEBUG="False")
        assert settings.DEBUG is False

        settings = BaseCoreSettings(DEBUG="1")
        assert settings.DEBUG is True

        settings = BaseCoreSettings(DEBUG="0")
        assert settings.DEBUG is False

    @pytest.mark.unit
    @pytest.mark.config
    def test_empty_string_validation(self):
        """Test validation behavior with empty strings."""
        # ENV field with empty string should fail
        with pytest.raises(ValidationError):
            BaseCoreSettings(ENV="")

        # LOG_LEVEL field with empty string should fail
        with pytest.raises(ValidationError):
            BaseCoreSettings(LOG_LEVEL="")

        # LOG_FORMAT field with empty string should fail
        with pytest.raises(ValidationError):
            BaseCoreSettings(LOG_FORMAT="")

    @pytest.mark.unit
    @pytest.mark.config
    def test_whitespace_only_validation(self):
        """Test validation behavior with whitespace-only strings."""
        # ENV field with whitespace-only should fail
        with pytest.raises(ValidationError):
            BaseCoreSettings(ENV="   ")

        # LOG_LEVEL field with whitespace-only should fail
        with pytest.raises(ValidationError):
            BaseCoreSettings(LOG_LEVEL="   ")

        # LOG_FORMAT field with whitespace-only should fail
        with pytest.raises(ValidationError):
            BaseCoreSettings(LOG_FORMAT="   ")

    @pytest.mark.unit
    @pytest.mark.config
    def test_special_characters_in_app_name(self):
        """Test APP_NAME field with special characters."""
        settings = BaseCoreSettings(APP_NAME="Trading-Chart_v1.0")
        assert settings.APP_NAME == "Trading-Chart_v1.0"

        settings = BaseCoreSettings(APP_NAME="TradingChart@2024")
        assert settings.APP_NAME == "TradingChart@2024"

        settings = BaseCoreSettings(APP_NAME="Trading Chart (Beta)")
        assert settings.APP_NAME == "Trading Chart (Beta)"

    @pytest.mark.unit
    @pytest.mark.config
    def test_unicode_characters_in_app_name(self):
        """Test APP_NAME field with Unicode characters."""
        settings = BaseCoreSettings(APP_NAME="交易圖表")
        assert settings.APP_NAME == "交易圖表"

        settings = BaseCoreSettings(APP_NAME="TradingChart™")
        assert settings.APP_NAME == "TradingChart™"

    @pytest.mark.unit
    @pytest.mark.config
    def test_boundary_values_string_length(self):
        """Test boundary values for string length."""
        # Test very long APP_NAME
        long_name = "A" * 1000
        settings = BaseCoreSettings(APP_NAME=long_name)
        assert settings.APP_NAME == long_name

        # Test valid long TIMEZONE name
        valid_long_timezone = "America/Argentina/Buenos_Aires"
        settings = BaseCoreSettings(TIMEZONE=valid_long_timezone)
        assert settings.TIMEZONE == valid_long_timezone

    @pytest.mark.unit
    @pytest.mark.config
    def test_case_insensitive_validation_edge_cases(self):
        """Test edge cases for case-insensitive validation."""
        # Test mixed case with spaces
        settings = BaseCoreSettings(ENV="  Dev  ")
        assert settings.ENV == "development"

        settings = BaseCoreSettings(LOG_LEVEL="  WaRnInG  ")
        assert settings.LOG_LEVEL == "WARNING"

        settings = BaseCoreSettings(LOG_FORMAT="  TeXt  ")
        assert settings.LOG_FORMAT == "txt"

    @pytest.mark.unit
    @pytest.mark.config
    def test_validator_direct_call_edge_cases(self):
        """Test direct validator calls with edge cases."""
        # Test ENV validator with edge cases
        env_validator = BaseCoreSettings.validate_env_case_insensitive
        assert env_validator("") == ""  # Empty string passes through
        assert env_validator("   ") == ""  # Whitespace-only gets stripped
        assert env_validator("DevElOpMeNt") == "development"  # Mixed case

        # Test LOG_LEVEL validator with edge cases
        log_level_validator = BaseCoreSettings.validate_log_level_case_insensitive
        assert log_level_validator("") == ""  # Empty string passes through
        assert log_level_validator("   ") == ""  # Whitespace-only gets stripped
        assert log_level_validator("dEbUg") == "DEBUG"  # Mixed case

        # Test LOG_FORMAT validator with edge cases
        log_format_validator = BaseCoreSettings.validate_log_format_case_insensitive
        assert log_format_validator("") == ""  # Empty string passes through
        assert log_format_validator("   ") == ""  # Whitespace-only gets stripped
        assert log_format_validator("JsOn") == "json"  # Mixed case

    @pytest.mark.unit
    @pytest.mark.config
    def test_multiple_field_validation_errors(self):
        """Test validation errors when multiple fields have invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            BaseCoreSettings(ENV="invalid_env", LOG_LEVEL="invalid_level", LOG_FORMAT="invalid_format")

        # Should contain errors for all invalid fields
        error_str = str(exc_info.value)
        assert "ENV" in error_str
        assert "LOG_LEVEL" in error_str
        assert "LOG_FORMAT" in error_str

    @pytest.mark.unit
    @pytest.mark.config
    def test_case_preservation_in_non_validated_fields(self):
        """Test that non-validated fields preserve case."""
        settings = BaseCoreSettings(APP_NAME="TradingChart", TIMEZONE="America/New_York")
        assert settings.APP_NAME == "TradingChart"  # Case preserved
        assert settings.TIMEZONE == "America/New_York"  # Case preserved

    @pytest.mark.unit
    @pytest.mark.config
    def test_validator_type_safety(self):
        """Test validators handle different input types safely."""
        # Test with None values
        env_validator = BaseCoreSettings.validate_env_case_insensitive
        assert env_validator(None) is None

        log_level_validator = BaseCoreSettings.validate_log_level_case_insensitive
        assert log_level_validator(None) is None

        log_format_validator = BaseCoreSettings.validate_log_format_case_insensitive
        assert log_format_validator(None) is None

        # Test with numeric values
        assert env_validator(123) == 123
        assert log_level_validator(456) == 456
        assert log_format_validator(789) == 789

        # Test with boolean values
        assert env_validator(True) is True
        assert log_level_validator(False) is False
        assert log_format_validator(True) is True

    @pytest.mark.unit
    @pytest.mark.config
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "APP_NAME": "TestApp",
                "ENV": "production",
                "DEBUG": "true",
                "LOG_LEVEL": "warning",
                "LOG_FORMAT": "json",
                "TIMEZONE": "Asia/Shanghai",
            },
        ):
            settings = BaseCoreSettings()
            assert settings.APP_NAME == "TestApp"
            assert settings.ENV == "production"
            assert settings.DEBUG is True
            assert settings.LOG_LEVEL == "WARNING"
            assert settings.LOG_FORMAT == "json"
            assert settings.TIMEZONE == "Asia/Shanghai"

    @pytest.mark.unit
    @pytest.mark.config
    def test_environment_variable_case_insensitive(self):
        """Test environment variable names are case-insensitive."""
        with patch.dict(
            os.environ,
            {
                "app_name": "TestApp",
                "env": "prod",
                "debug": "false",
                "log_level": "debug",
                "log_format": "structured",
                "timezone": "Europe/London",
            },
        ):
            settings = BaseCoreSettings()
            assert settings.APP_NAME == "TestApp"
            assert settings.ENV == "production"
            assert settings.DEBUG is False
            assert settings.LOG_LEVEL == "DEBUG"
            assert settings.LOG_FORMAT == "json"
            assert settings.TIMEZONE == "Europe/London"

    @pytest.mark.unit
    @pytest.mark.config
    def test_environment_variable_override_defaults(self):
        """Test environment variables override default values."""
        # Test with some env vars set
        with patch.dict(os.environ, {"ENV": "staging", "LOG_LEVEL": "error"}):
            settings = BaseCoreSettings()
            assert settings.ENV == "staging"  # Overridden
            assert settings.LOG_LEVEL == "ERROR"  # Overridden
            assert settings.APP_NAME == "TradingChart"  # Default
            assert settings.DEBUG is False  # Default
            assert settings.LOG_FORMAT == "txt"  # Default

    @pytest.mark.unit
    @pytest.mark.config
    def test_dotenv_file_loading(self):
        """Test loading configuration from .env file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("""
APP_NAME=DotEnvApp
ENV=production
DEBUG=true
LOG_LEVEL=critical
LOG_FORMAT=json
TIMEZONE=America/New_York
""")

            # Change to temp directory so .env is found
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                settings = BaseCoreSettings()
                assert settings.APP_NAME == "DotEnvApp"
                assert settings.ENV == "production"
                assert settings.DEBUG is True
                assert settings.LOG_LEVEL == "CRITICAL"
                assert settings.LOG_FORMAT == "json"
                assert settings.TIMEZONE == "America/New_York"
            finally:
                os.chdir(original_cwd)

    @pytest.mark.unit
    @pytest.mark.config
    def test_dotenv_case_insensitive_values(self):
        """Test .env file values are processed case-insensitively."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text(
                """
                ENV=STAGING
                LOG_LEVEL=info
                LOG_FORMAT=TXT
                """
            )

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                settings = BaseCoreSettings()
                assert settings.ENV == "staging"
                assert settings.LOG_LEVEL == "INFO"
                assert settings.LOG_FORMAT == "txt"
            finally:
                os.chdir(original_cwd)

    @pytest.mark.unit
    @pytest.mark.config
    def test_environment_variable_precedence_over_dotenv(self):
        """Test environment variables take precedence over .env file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("""
APP_NAME=DotEnvApp
ENV=development
DEBUG=false
""")

            with patch.dict(os.environ, {"ENV": "production", "DEBUG": "true"}):
                original_cwd = os.getcwd()
                try:
                    os.chdir(tmp_dir)
                    settings = BaseCoreSettings()
                    assert settings.APP_NAME == "DotEnvApp"  # From .env
                    assert settings.ENV == "production"  # From env var (overrides .env)
                    assert settings.DEBUG is True  # From env var (overrides .env)
                finally:
                    os.chdir(original_cwd)

    @pytest.mark.unit
    @pytest.mark.config
    def test_malformed_dotenv_file_handling(self):
        """Test handling of malformed .env files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text("""
# Comment line
APP_NAME=ValidApp
INVALID_LINE_WITHOUT_EQUALS
ENV=production
EMPTY_VALUE=
LOG_LEVEL=INFO
""")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                # Should not raise an exception, just ignore malformed lines
                settings = BaseCoreSettings()
                assert settings.APP_NAME == "ValidApp"
                assert settings.ENV == "production"
                assert settings.LOG_LEVEL == "INFO"
            finally:
                os.chdir(original_cwd)

    @pytest.mark.unit
    @pytest.mark.config
    def test_unicode_in_dotenv_file(self):
        """Test Unicode characters in .env file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_file = Path(tmp_dir) / ".env"
            env_file.write_text(
                """
APP_NAME=交易圖表
TIMEZONE=Asia/Shanghai
""",
                encoding="utf-8",
            )

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                settings = BaseCoreSettings()
                assert settings.APP_NAME == "交易圖表"
                assert settings.TIMEZONE == "Asia/Shanghai"
            finally:
                os.chdir(original_cwd)

    @pytest.mark.unit
    @pytest.mark.config
    def test_extra_fields_ignored(self):
        """Test that extra fields in environment/dotenv are ignored."""
        with patch.dict(
            os.environ, {"APP_NAME": "TestApp", "UNKNOWN_FIELD": "should_be_ignored", "ANOTHER_UNKNOWN": "also_ignored"}
        ):
            # Should not raise an exception
            settings = BaseCoreSettings()
            assert settings.APP_NAME == "TestApp"
            assert not hasattr(settings, "UNKNOWN_FIELD")
            assert not hasattr(settings, "ANOTHER_UNKNOWN")

    @pytest.mark.unit
    @pytest.mark.config
    def test_settings_immutability(self):
        """Test that settings object behaves as expected with field assignment."""
        settings = BaseCoreSettings()
        original_app_name = settings.APP_NAME

        # Pydantic v2 allows field assignment by default, but we can test the behavior
        # If frozen=True was set in model_config, this would raise an exception
        # For now, we'll just test that the assignment works as expected
        settings.APP_NAME = "Modified"
        assert settings.APP_NAME == "Modified"

        # Reset for other tests
        settings.APP_NAME = original_app_name

    @pytest.mark.unit
    @pytest.mark.config
    def test_settings_dict_export(self):
        """Test converting settings to dictionary."""
        settings = BaseCoreSettings(
            APP_NAME="TestApp", ENV="production", DEBUG=True, LOG_LEVEL="WARNING", LOG_FORMAT="json", TIMEZONE="UTC"
        )

        settings_dict = settings.model_dump()
        expected_dict = {
            "APP_NAME": "TestApp",
            "ENV": "production",
            "DEBUG": True,
            "LOG_LEVEL": "WARNING",
            "LOG_FORMAT": "json",
            "TIMEZONE": "UTC",
        }

        assert settings_dict == expected_dict

    @pytest.mark.unit
    @pytest.mark.config
    def test_settings_json_export(self):
        """Test converting settings to JSON string."""
        settings = BaseCoreSettings(APP_NAME="TestApp", ENV="production", DEBUG=True)

        json_str = settings.model_dump_json()
        assert isinstance(json_str, str)
        assert "TestApp" in json_str
        assert "production" in json_str
        assert "true" in json_str.lower()

    @pytest.mark.unit
    @pytest.mark.config
    def test_field_info_access(self):
        """Test accessing field metadata and descriptions."""
        # Access model_fields from class, not instance (avoids deprecation warning)
        fields = BaseCoreSettings.model_fields
        assert "APP_NAME" in fields
        assert "ENV" in fields
        assert "DEBUG" in fields
        assert "LOG_LEVEL" in fields
        assert "LOG_FORMAT" in fields
        assert "TIMEZONE" in fields

        # Test field descriptions
        assert fields["APP_NAME"].description is not None
        assert fields["ENV"].description is not None
        assert "identification" in fields["APP_NAME"].description.lower()
        assert "environment" in fields["ENV"].description.lower()

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_validation_whitespace_edge_cases(self):
        """Test TIMEZONE field handles whitespace edge cases specifically."""
        # Test empty string after stripping
        with pytest.raises(ValidationError) as exc_info:
            BaseCoreSettings(TIMEZONE="   ")
        assert "Invalid timezone" in str(exc_info.value)

        # Test with newlines and tabs
        settings = BaseCoreSettings(TIMEZONE="\t\nUTC\n\t")
        assert settings.TIMEZONE == "UTC"

    @pytest.mark.unit
    @pytest.mark.config  
    def test_timezone_validation_complex_iana_names(self):
        """Test TIMEZONE field with complex IANA timezone names."""
        complex_timezones = [
            "America/North_Dakota/New_Salem",
            "America/Kentucky/Louisville", 
            "Pacific/Pitcairn",
            "Antarctica/McMurdo",
            "Europe/Busingen",
        ]
        
        for tz in complex_timezones:
            settings = BaseCoreSettings(TIMEZONE=tz)
            assert settings.TIMEZONE == tz

    @pytest.mark.unit
    @pytest.mark.config
    def test_env_validator_with_special_characters(self):
        """Test ENV field validation with special characters."""
        # These should all fail validation as they're not valid environments
        invalid_envs = ["dev-1", "prod_2", "staging@test", "development#1"]
        
        for invalid_env in invalid_envs:
            with pytest.raises(ValidationError):
                BaseCoreSettings(ENV=invalid_env)

    @pytest.mark.unit
    @pytest.mark.config
    def test_log_level_with_numeric_strings(self):
        """Test LOG_LEVEL field with numeric strings (should fail)."""
        invalid_levels = ["1", "2", "3", "0", "10"]
        
        for invalid_level in invalid_levels:
            with pytest.raises(ValidationError):
                BaseCoreSettings(LOG_LEVEL=invalid_level)

    @pytest.mark.unit
    @pytest.mark.config  
    def test_validator_chaining_behavior(self):
        """Test that multiple validators work together correctly."""
        # Test that field validation occurs before model validation
        settings = BaseCoreSettings(
            ENV="  PROD  ",
            LOG_LEVEL="  warning  ", 
            LOG_FORMAT="  STRUCTURED  ",
            TIMEZONE="  america/new_york  "
        )
        
        assert settings.ENV == "production"
        assert settings.LOG_LEVEL == "WARNING"
        assert settings.LOG_FORMAT == "json"
        assert settings.TIMEZONE == "America/New_York"

    @pytest.mark.unit
    @pytest.mark.config
    def test_model_config_validation(self):
        """Test model configuration settings."""
        # Test that extra fields are ignored as configured
        settings = BaseCoreSettings(_env_file="nonexistent.env", EXTRA_FIELD="ignored")
        
        # Should create successfully with defaults (extra field ignored)
        assert settings.APP_NAME == "TradingChart"
        assert not hasattr(settings, "EXTRA_FIELD")

    @pytest.mark.unit
    @pytest.mark.config
    def test_timezone_case_edge_cases(self):
        """Test TIMEZONE case normalization edge cases."""
        # Test deep nested timezones with case normalization
        test_cases = [
            ("america/argentina/buenos_aires", "America/Argentina/Buenos_Aires"),
            ("america/north_dakota/new_salem", "America/North_Dakota/New_Salem"),
            ("pacific/marquesas", "Pacific/Marquesas"),
        ]
        
        for input_tz, expected_tz in test_cases:
            settings = BaseCoreSettings(TIMEZONE=input_tz)
            assert settings.TIMEZONE == expected_tz
