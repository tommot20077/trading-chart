# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased - 2025-07-17 [9c372b](https://github.com/tommot20077/trading-chart/commit/9c372bd52d2fc6a2ef6118e2e08a478ce4e548e9)

### Added
- **Dependency Review Workflow**: Added GitHub Actions workflow for dependency security and license review
- **Complete Core System Implementation**: Added full implementation of all core interfaces with memory and noop providers
- **Authentication System**: Complete in-memory authentication with user management, password hashing, and token-based auth
- **Event Processing System**: Event bus, serialization, and middleware pipeline with comprehensive event handling
- **Storage System**: Time-series repository, event storage, and metadata management with full CRUD operations
- **Middleware Pipeline**: Request/response middleware system with context propagation and error handling
- **Observability System**: Notification handling and monitoring capabilities
- **Rate Limiting**: In-memory rate limiting with configurable policies
- **Comprehensive Testing**: Added unit, integration, and contract tests for all components
- **Type Safety**: Complete MyPy integration with strict type checking across all modules

### Changed
- **Python Version Support**: Updated from Python 3.10-3.13 to 3.11-3.13 support
- **Project Structure**: Moved handler registry from interfaces to components package
- **Exception Handling**: Enhanced exception hierarchy with specific event and storage exceptions
- **Test Configuration**: Added pytest-timeout and enhanced test markers for better test organization
- **CI/CD Pipeline**: Updated GitHub Actions to support Python 3.11-3.13 versions

### Removed
- **AI Documentation**: Removed ai-docs/SUMMARY.md and ai-docs/WORKFLOWS.md files
- **Benchmark Baselines**: Removed benchmarks/baselines directory structure
- **Python 3.10 Support**: Dropped Python 3.10 support in favor of newer versions

### Fixed
- **Type Checking**: Resolved type annotation issues with disable_error_code for unchecked annotations
- **Test Timeout**: Added proper timeout handling for concurrent tests
- **Format Check**: Fixed format check command in poethepoet configuration

### Breaking Changes
- **Minimum Python Version**: Now requires Python 3.11 or higher
- **Architecture**: Complete restructuring of core implementations with new interface contracts
- **Authentication**: New authentication system requiring token-based authentication
- **Event System**: New event processing architecture with middleware support

### Performance Improvements
- **Memory Management**: Optimized in-memory implementations with better data structures
- **Event Processing**: Efficient event serialization and deserialization
- **Rate Limiting**: High-performance in-memory rate limiting with minimal overhead
- **Concurrent Operations**: Thread-safe implementations for all core components

### Documentation
- **Contract Testing**: Added comprehensive contract test documentation
- **Integration Testing**: Enhanced integration test coverage with real-world scenarios
- **API Documentation**: Complete API documentation for all new interfaces and implementations

---

## Unreleased - 2025-07-15 [864bf9](https://github.com/tommot20077/trading-chart/commit/864bf9df6d2bdbd54432e36ea14c49127af2ed8a)

### Added
- **Authentication System**: Full authentication interfaces including authenticator, authorizer, and token manager
- **Data Models**: Comprehensive data models for trading systems (Kline, Trade, BaseEvent)
- **Event System**: Complete event architecture with priority-based handling and storage
- **Storage Interfaces**: Time-series database interfaces with generic repository patterns
- **Network Models**: Network-related enums and models for system communication
- **Exception Handling**: Comprehensive exception hierarchy for error management
- **Data Validation**: Robust data validation with Pydantic integration
- **Type Safety**: Complete MyPy type checking integration
- **Github Actions CI/CD**: Full CI/CD pipeline with automated testing

### Changed
- **Architecture Design**: Evolved from simple configuration to comprehensive trading system architecture
- **Data Layer**: Introduced sophisticated data models with validation and serialization
- **Event System**: Implemented priority-based event handling with subscription management
- **Storage Layer**: Added generic repository patterns for time-series data management

### Fixed
- **EventPriority Type Safety**: Fixed type compatibility issues between EventPriority and Pydantic
- **Method Signatures**: Resolved method signature conflicts in repository inheritance
- **Type Annotations**: Added missing return type annotations for complete type safety
- **Data Validation**: Fixed timezone handling and price/volume validation edge cases

### Breaking Changes
- **Architecture Shift**: Moved from simple configuration to comprehensive trading system architecture
- **Data Models**: Introduced strict data validation with Pydantic models
- **Event System**: Implemented new event-driven architecture with priority handling
- **Storage Layer**: Added repository pattern requiring interface implementations

### Performance Improvements
- **Data Validation**: Efficient Pydantic validation with field-level optimizations
- **Event Processing**: Priority-based event handling with O(log n) complexity
- **Type Checking**: Complete MyPy integration with zero type errors
- **Memory Usage**: Optimized object creation and memory footprint

### Documentation
- Added comprehensive API documentation for all core interfaces
- Documented data model schemas and validation rules
- Enhanced event system architecture documentation
- Added storage interface usage examples
- Updated development workflows with new architecture patterns

---

## Unreleased - 2025-07-14 [127ea7](https://github.com/tommot20077/trading-chart/commit/127ea7d92325975dd0b14f8372e9bdc48685348c)

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