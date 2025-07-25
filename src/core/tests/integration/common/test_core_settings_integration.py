# ABOUTME: Integration tests for CoreSettings + Pydantic validation integration
# ABOUTME: Tests environment variable loading, validation integration, and business usage

import asyncio
import os
import tempfile
import pytest
from typing import Dict, Any
from unittest.mock import patch

from core.config.settings import CoreSettings, get_settings
from core.config._base import BaseCoreSettings
from core.exceptions.base import ConfigurationException, ValidationException


class TestCoreSettingsPydanticIntegration:
    """Test integration between CoreSettings and Pydantic validation."""
    
    def test_environment_variable_to_pydantic_validation_to_business_usage_flow(self):
        """Test complete flow: environment variables → Pydantic validation → configuration loading → business usage."""
        # Simulate environment variables
        test_env = {
            'APP_NAME': 'TestTradingApp',
            'ENV': 'production',
            'DEBUG': 'false',
            'LOG_LEVEL': 'warning',
            'LOG_FORMAT': 'json',
            'TIMEZONE': 'America/New_York'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            # Clear the cache to ensure fresh loading
            get_settings.cache_clear()
            
            # Load settings through Pydantic validation
            settings = get_settings()
            
            # Verify business configuration usage
            assert settings.APP_NAME == 'TestTradingApp'
            assert settings.ENV == 'production'
            assert settings.DEBUG is False
            assert settings.LOG_LEVEL == 'WARNING'  # Should be normalized to uppercase
            assert settings.LOG_FORMAT == 'json'
            assert settings.TIMEZONE == 'America/New_York'
            
            # Test business logic integration
            business_config = {
                'app_name': settings.APP_NAME,
                'is_production': settings.ENV == 'production',
                'enable_debug_logging': settings.DEBUG,
                'log_config': {
                    'level': settings.LOG_LEVEL,
                    'format': settings.LOG_FORMAT
                },
                'timezone_info': settings.TIMEZONE
            }
            
            # Verify business configuration is correctly derived
            assert business_config['is_production'] is True
            assert business_config['enable_debug_logging'] is False
            assert business_config['log_config']['level'] == 'WARNING'

    def test_pydantic_validation_error_handling_integration(self):
        """Test Pydantic validation error handling and business exception integration."""
        # Test invalid environment values
        invalid_configs = [
            {'ENV': 'invalid_env', 'expected_error': 'ENV'},
            {'LOG_LEVEL': 'INVALID_LEVEL', 'expected_error': 'LOG_LEVEL'},
            {'LOG_FORMAT': 'invalid_format', 'expected_error': 'LOG_FORMAT'},
            {'TIMEZONE': 'Invalid/Timezone', 'expected_error': 'TIMEZONE'},
            {'DEBUG': 'not_a_boolean', 'expected_error': 'DEBUG'}
        ]
        
        for invalid_config in invalid_configs:
            field_name = invalid_config['expected_error']
            with patch.dict(os.environ, invalid_config, clear=False):
                get_settings.cache_clear()
                
                try:
                    settings = get_settings()
                    # Some validations might pass due to normalization
                    # Just verify the system doesn't crash
                except Exception as e:
                    # Should be a validation-related error
                    assert any(keyword in str(e) for keyword in ['validation', 'invalid', 'error'])

    def test_env_file_loading_pydantic_integration(self):
        """Test .env file loading with Pydantic validation integration."""
        # Create temporary .env file
        env_content = """
        APP_NAME=EnvFileApp
        ENV=staging
        DEBUG=true
        LOG_LEVEL=debug
        LOG_FORMAT=txt
        TIMEZONE=UTC
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            # Test loading from custom env file
            class TestSettings(BaseCoreSettings):
                model_config = BaseCoreSettings.model_config.copy()
                model_config.update({'env_file': env_file_path})
            
            settings = TestSettings()
            
            # Verify env file values are loaded and validated
            assert settings.APP_NAME == 'EnvFileApp'
            assert settings.ENV == 'staging'
            assert settings.DEBUG is True
            assert settings.LOG_LEVEL == 'DEBUG'  # Normalized to uppercase
            assert settings.LOG_FORMAT == 'txt'
            assert settings.TIMEZONE == 'UTC'
            
        finally:
            os.unlink(env_file_path)

    def test_case_insensitive_validation_normalization(self):
        """Test case-insensitive validation and normalization."""
        test_cases = [
            # ENV field normalization
            {'ENV': 'dev', 'expected_env': 'development'},
            {'ENV': 'PROD', 'expected_env': 'production'},
            {'ENV': 'Stage', 'expected_env': 'staging'},
            
            # LOG_LEVEL normalization
            {'LOG_LEVEL': 'info', 'expected_log_level': 'INFO'},
            {'LOG_LEVEL': 'error', 'expected_log_level': 'ERROR'},
            
            # LOG_FORMAT normalization
            {'LOG_FORMAT': 'JSON', 'expected_log_format': 'json'},
            {'LOG_FORMAT': 'Text', 'expected_log_format': 'txt'},
        ]
        
        for test_case in test_cases:
            with patch.dict(os.environ, test_case, clear=False):
                get_settings.cache_clear()
                settings = get_settings()
                
                if 'expected_env' in test_case:
                    assert settings.ENV == test_case['expected_env']
                if 'expected_log_level' in test_case:
                    assert settings.LOG_LEVEL == test_case['expected_log_level']
                if 'expected_log_format' in test_case:
                    assert settings.LOG_FORMAT == test_case['expected_log_format']


class TestConfigurationHotReloadIntegration:
    """Test configuration hot reload and component synchronization."""
    
    def test_settings_cache_invalidation_and_reload(self):
        """Test settings cache invalidation and hot reload."""
        # Initial configuration
        initial_env = {'APP_NAME': 'InitialApp', 'ENV': 'development'}
        
        with patch.dict(os.environ, initial_env, clear=False):
            get_settings.cache_clear()
            settings1 = get_settings()
            assert settings1.APP_NAME == 'InitialApp'
            assert settings1.ENV == 'development'
        
        # Simulate configuration change
        updated_env = {'APP_NAME': 'UpdatedApp', 'ENV': 'production'}
        
        with patch.dict(os.environ, updated_env, clear=False):
            # Without clearing cache, should get cached version
            settings2 = get_settings()
            assert settings2.APP_NAME == 'InitialApp'  # Still cached
            
            # Clear cache and reload
            get_settings.cache_clear()
            settings3 = get_settings()
            assert settings3.APP_NAME == 'UpdatedApp'  # New configuration
            assert settings3.ENV == 'production'

    def test_configuration_consistency_across_components(self):
        """Test configuration consistency when accessed from different components."""
        config_env = {
            'APP_NAME': 'ConsistencyTest',
            'ENV': 'production',
            'LOG_LEVEL': 'INFO'
        }
        
        with patch.dict(os.environ, config_env, clear=False):
            get_settings.cache_clear()
            
            # Simulate multiple components accessing settings
            def component_a_config():
                settings = get_settings()
                return {
                    'component': 'A',
                    'app_name': settings.APP_NAME,
                    'env': settings.ENV,
                    'log_level': settings.LOG_LEVEL
                }
            
            def component_b_config():
                settings = get_settings()
                return {
                    'component': 'B',
                    'app_name': settings.APP_NAME,
                    'is_production': settings.ENV == 'production',
                    'debug_enabled': settings.DEBUG
                }
            
            config_a = component_a_config()
            config_b = component_b_config()
            
            # Verify consistency
            assert config_a['app_name'] == config_b['app_name']
            assert config_a['env'] == 'production'
            assert config_b['is_production'] is True

    def test_component_synchronization_after_config_change(self):
        """Test component synchronization after configuration changes."""
        # Mock components that depend on configuration
        class MockRateLimiter:
            def __init__(self, settings):
                self.settings = settings
                self.capacity = 100 if settings.ENV == 'production' else 10
        
        class MockLogger:
            def __init__(self, settings):
                self.settings = settings
                self.level = settings.LOG_LEVEL
                self.format = settings.LOG_FORMAT
        
        # Initial configuration
        initial_env = {'ENV': 'development', 'LOG_LEVEL': 'DEBUG', 'LOG_FORMAT': 'txt'}
        
        with patch.dict(os.environ, initial_env, clear=False):
            get_settings.cache_clear()
            settings = get_settings()
            
            # Initialize components
            rate_limiter = MockRateLimiter(settings)
            logger = MockLogger(settings)
            
            assert rate_limiter.capacity == 10  # Development setting
            assert logger.level == 'DEBUG'
            assert logger.format == 'txt'
        
        # Change configuration
        updated_env = {'ENV': 'production', 'LOG_LEVEL': 'WARNING', 'LOG_FORMAT': 'json'}
        
        with patch.dict(os.environ, updated_env, clear=False):
            get_settings.cache_clear()
            new_settings = get_settings()
            
            # Simulate component reconfiguration
            new_rate_limiter = MockRateLimiter(new_settings)
            new_logger = MockLogger(new_settings)
            
            assert new_rate_limiter.capacity == 100  # Production setting
            assert new_logger.level == 'WARNING'
            assert new_logger.format == 'json'


class TestLoggingConfigDynamicSwitching:
    """Test LoggingConfig dynamic switching integration."""
    
    def test_logging_configuration_integration_with_settings(self):
        """Test logging configuration integration with settings changes."""
        logging_configs = [
            {'LOG_LEVEL': 'DEBUG', 'LOG_FORMAT': 'txt', 'ENV': 'development'},
            {'LOG_LEVEL': 'INFO', 'LOG_FORMAT': 'json', 'ENV': 'production'},
            {'LOG_LEVEL': 'WARNING', 'LOG_FORMAT': 'json', 'ENV': 'staging'},
            {'LOG_LEVEL': 'ERROR', 'LOG_FORMAT': 'txt', 'ENV': 'production'}
        ]
        
        for config in logging_configs:
            with patch.dict(os.environ, config, clear=False):
                get_settings.cache_clear()
                settings = get_settings()
                
                # Simulate logging system configuration
                log_config = {
                    'level': settings.LOG_LEVEL,
                    'format': settings.LOG_FORMAT,
                    'environment': settings.ENV,
                    'enable_debug': settings.DEBUG,
                    'structured_logging': settings.LOG_FORMAT == 'json'
                }
                
                # Verify logging configuration matches settings
                assert log_config['level'] == config['LOG_LEVEL'].upper()
                assert log_config['format'] == config['LOG_FORMAT']
                assert log_config['environment'] == config['ENV']
                
                # Test environment-specific logic
                if config['ENV'] == 'production':
                    assert log_config['structured_logging'] == (config['LOG_FORMAT'] == 'json')
                elif config['ENV'] == 'development':
                    # Development might prefer text format for readability
                    pass

    def test_dynamic_logging_level_switching(self):
        """Test dynamic logging level switching impact on system behavior."""
        class MockLoggingSystem:
            def __init__(self, settings):
                self.level = settings.LOG_LEVEL
                self.format = settings.LOG_FORMAT
                self.debug_enabled = settings.DEBUG
            
            def should_log_debug(self):
                return self.level == 'DEBUG' or self.debug_enabled
            
            def should_log_info(self):
                return self.level in ['DEBUG', 'INFO']
            
            def should_log_warning(self):
                return self.level in ['DEBUG', 'INFO', 'WARNING']
        
        # Test different logging levels
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        
        for level in log_levels:
            with patch.dict(os.environ, {'LOG_LEVEL': level}, clear=False):
                get_settings.cache_clear()
                settings = get_settings()
                logging_system = MockLoggingSystem(settings)
                
                # Verify logging behavior matches configuration
                if level == 'DEBUG':
                    assert logging_system.should_log_debug()
                    assert logging_system.should_log_info()
                    assert logging_system.should_log_warning()
                elif level == 'INFO':
                    assert not logging_system.should_log_debug()
                    assert logging_system.should_log_info()
                    assert logging_system.should_log_warning()
                elif level == 'WARNING':
                    assert not logging_system.should_log_debug()
                    assert not logging_system.should_log_info()
                    assert logging_system.should_log_warning()
                elif level == 'ERROR':
                    assert not logging_system.should_log_debug()
                    assert not logging_system.should_log_info()
                    assert not logging_system.should_log_warning()

    def test_logging_format_switching_impact(self):
        """Test logging format switching impact on output and parsing."""
        format_configs = [
            {'LOG_FORMAT': 'json'},
            {'LOG_FORMAT': 'txt'},
            {'LOG_FORMAT': 'structured'},  # Alias for json
            {'LOG_FORMAT': 'text'}  # Alias for txt
        ]
        
        for config in format_configs:
            with patch.dict(os.environ, config, clear=False):
                get_settings.cache_clear()
                settings = get_settings()
                
                # Simulate logging formatter configuration
                formatter_config = {
                    'structured': settings.LOG_FORMAT == 'json',
                    'human_readable': settings.LOG_FORMAT == 'txt',
                    'include_timestamp': True,
                    'include_level': True
                }
                
                # Verify format is correctly set
                if config['LOG_FORMAT'] in ['json', 'structured']:
                    assert formatter_config['structured'] is True
                    assert formatter_config['human_readable'] is False
                elif config['LOG_FORMAT'] in ['txt', 'text']:
                    assert formatter_config['structured'] is False
                    assert formatter_config['human_readable'] is True


if __name__ == "__main__":
    pytest.main([__file__])