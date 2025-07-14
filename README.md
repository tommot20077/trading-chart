# TradingChart - Trading Chart System

Modern trading data infrastructure with clean architecture design.

## Project Overview

- **Purpose**: Modern trading data infrastructure with clean architecture
- **Language**: Python 3.12+
- **Architecture**: Four-layer monorepo with dependency injection
- **Package Manager**: uv (strictly enforced)
- **Testing Framework**: pytest with multi-layered testing strategy
- **Code Quality**: ruff (linting), mypy (type checking), automated quality gates

## Architecture Layers

### Core Layer (`src/core/`)
- **Purpose**: Defines contracts and provides default in-memory implementations
- **Principle**: Minimal, foundational dependencies
- **Contains**: Interfaces, models, base implementations
- **Dependencies**: None (foundation layer)

### Libs Layer (`src/libs/`)
- **Purpose**: Reusable internal business logic
- **Principle**: Internal, business-oriented functionality
- **Contains**: Authentication, utilities, business services
- **Dependencies**: core only

### Integrations Layer (`src/integrations/`)
- **Purpose**: External infrastructure interactions
- **Principle**: External, infrastructure-oriented
- **Contains**: Database adapters, external API clients, messaging
- **Dependencies**: core only

### Apps Layer (`src/apps/`)
- **Purpose**: Application assembly and specific business logic
- **Principle**: Assembly & specific logic
- **Contains**: FastAPI applications, CLI tools, workers
- **Dependencies**: core, libs, integrations

## Development Commands

### Package Management
- **MANDATORY**: Always use `uv` for all package management operations
- `uv add <dependency>`: Add dependency to specific package
- `uv sync`: Sync workspace dependencies
- `uv run <command>`: Execute commands in virtual environment

### Testing Strategy
- **Multi-layered testing**: Unit → Integration → Contract → E2E → Benchmark
- **Test markers**: Use `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.benchmark`, etc.
- **Time control**: Use `time_machine` directly (NO custom wrappers)
- **Coverage**: Minimum 80% coverage required

### Quality Assurance
- **Automated gates**: All commits must pass quality checks
- **Tools**: ruff (linting), mypy (type checking), pytest (testing)
- **Pre-commit hooks**: Automatically enforce standards
- **No bypass**: NEVER use `--no-verify` when committing

## Benchmark Testing

### Using pytest-benchmark
```bash
# Run all benchmark tests
uv run pytest src/core/tests/benchmark/ --benchmark-only

# Save performance baseline
python scripts/run_benchmarks.py --save-baseline

# Compare against baseline
python scripts/run_benchmarks.py --compare baseline

# Generate JSON report
python scripts/run_benchmarks.py --format json
```

### Current Performance Benchmarks
- **Singleton Access**: ~77ns (13M ops/sec)
- **Basic Validators**: ~400μs (2.5K ops/sec)
- **TIMEZONE Validation**: ~400μs (standard) / ~10.8ms (case normalization)
- **Full Object Creation**: ~450μs (2.2K ops/sec)

## Code Standards

### File Structure
- **ABOUTME comments**: All files must start with 2-line ABOUTME comment
  ```python
  # ABOUTME: Brief description of what this file does
  # ABOUTME: Additional context or purpose
  ```
- **Interface Segregation**: Follow ISP principles for clean contracts
- **Dependency Injection**: Use dependency-injector for IoC

### Development Principles
- **Clean Architecture**: Respect layer boundaries and dependency rules
- **SOLID Principles**: Especially Single Responsibility and Interface Segregation
- **Test-Driven Development**: Write tests first, then implementation
- **Evidence-based decisions**: Provide test results and documentation
- **Incremental changes**: Make smallest reasonable changes

## Configuration System

### Enhanced Validators
```python
from core.config._base import BaseCoreSettings

# These all work correctly and normalize:
settings = BaseCoreSettings(
    ENV="PROD",              # → "production"
    LOG_LEVEL="debug",       # → "DEBUG"  
    LOG_FORMAT="structured", # → "json"
    TIMEZONE="america/new_york"  # → "America/New_York"
)
```

### Supported Aliases
- **ENV**: `dev`/`develop` → `development`, `prod` → `production`, `stage` → `staging`
- **LOG_FORMAT**: `structured` → `json`, `text` → `console`
- **TIMEZONE**: Full IANA timezone validation with case normalization

## Directory Structure

### Core Layer (`src/core/`)
- `core/interfaces/`: Abstract contracts and protocols
- `core/models/`: Data models (market_data, events, base)
- `core/implementations/`: Default in-memory implementations
- `core/config/`: Configuration management
- `tests/`: Comprehensive test suites

### Test Directories
- `tests/unit/`: Unit tests
- `tests/integration/`: Integration tests
- `tests/benchmark/`: Performance tests
- `tests/contract/`: Contract tests

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repo_url> && cd trading-chart

# Install all dependencies
uv sync --dev

# Install core package as editable
uv add --editable src/core
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test types
uv run pytest -m unit
uv run pytest -m benchmark

# Run code quality checks
uv run ruff check .
uv run mypy .
```

### Development Workflow
```bash
# Check code format
uv run ruff format --check .

# Fix code format
uv run ruff format .

# Run all quality checks
uv run ruff check . && uv run mypy . && uv run pytest
```

## Security Framework

- **Input Validation**: Comprehensive sanitization at all entry points
- **Authentication**: JWT-based authentication with role-based access
- **Audit Trail**: Complete security event logging
- **Encryption**: Sensitive data encryption at rest and in transit
- **Security Testing**: Dedicated security test suites

## Quality Assurance

- **Automated Testing**: Multi-layered test strategy with 80%+ coverage
- **Code Quality Gates**: Automated ruff, mypy, and pytest checks
- **Performance Monitoring**: Built-in performance tracking and alerting
- **Security Scanning**: Regular security vulnerability assessments
- **Documentation Standards**: Comprehensive inline and external documentation

## Contributing Guidelines

### Git Workflow
- **Branch Strategy**: Feature branches with PR-based integration
- **Commit Standards**: Conventional commits with clear, descriptive messages
- **Quality Gates**: All PRs must pass automated quality checks
- **Code Review**: Mandatory peer review before merging

### Testing Workflow
1. **Write Tests First**: TDD approach for all new features
2. **Multi-layer Testing**: Unit → Integration → Contract → E2E → Benchmark
3. **Quality Gates**: 80%+ coverage required for all layers
4. **Time Control**: Use `time_machine` directly for deterministic tests

## Support

- **Issue Reporting**: Use GitHub Issues
- **Feature Requests**: Through GitHub Discussions
- **Documentation**: Check `ai-docs/` directory
- **Examples**: See `examples/` directory (coming soon)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project is under active development. APIs may change until reaching version 1.0.0.
