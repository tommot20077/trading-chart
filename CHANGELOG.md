# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased - 2025-07-15]

### Init
- Initial project setup with monorepo architecture
- Core configuration system with BaseCoreSettings
- Four-layer architecture (Core, Libs, Integrations, Apps)
- Comprehensive testing strategy (Unit, Integration, E2E)
- Development tooling with uv package manager
- Code quality tools (ruff, mypy, pytest)
- Dependency injection framework setup

### Added
- **Enhanced Configuration Validators**: Case-insensitive field validators for ENV, LOG_LEVEL, LOG_FORMAT, and TIMEZONE fields
- **Environment Aliases**: Support for common environment aliases (dev→development, prod→production, stage→staging)
- **TIMEZONE Validation**: Comprehensive IANA timezone validation using Python's zoneinfo module with case normalization
- **Professional Benchmark Testing**: Replaced custom performance tests with pytest-benchmark framework
- **Comprehensive Test Suite**: Added complete benchmark test suite for validators and settings performance
- **Benchmark Scripts**: Included performance testing scripts with baseline comparison capabilities
- **Documentation Updates**: Enhanced project documentation with benchmark testing workflows
- **Multi-language Documentation**: Added Traditional Chinese documentation support

### Changed
- **Performance Testing Framework**: Migrated from custom time measurement to pytest-benchmark for accurate performance analysis
- **Configuration Validation**: Enhanced validation logic with better error messages and case handling
- **Test Structure**: Reorganized test structure to include benchmark testing alongside unit and integration tests

### Fixed
- **Test Isolation**: Fixed cache-related test isolation issues in benchmark tests
- **TIMEZONE Case Handling**: Improved timezone validation to handle various case formats correctly

### Breaking Changes
- **TIMEZONE Validation**: TIMEZONE field now strictly validates against IANA timezone database. Invalid timezone strings will raise validation errors.

### Performance Improvements
- **Singleton Access**: Settings singleton access optimized to ~77ns (13M ops/sec)
- **Validator Performance**: Most validators perform at ~400μs (2.5K ops/sec)
- **Object Creation**: Complete settings object creation at ~450μs (2.2K ops/sec)

### Documentation
- Added comprehensive benchmark testing guide
- Updated development workflows with performance testing procedures
- Enhanced project summary with benchmark testing information
- Added Traditional Chinese documentation

### Technical Details
- **Dependencies**: Added pytest-benchmark and psutil for performance testing
- **Test Coverage**: Maintained 80%+ test coverage across all layers
- **Code Quality**: All changes pass ruff, mypy, and pytest quality gates
- **Architecture**: Maintained clean architecture principles with proper layer separation
