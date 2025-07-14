# Project Summary & Cheatsheet

This document provides a high-level overview of the project architecture and a list of common commands for daily
development.

---

## 1. Core Architecture

The project uses a four-layer, dependency-injected, monorepo architecture.

| Layer            | Responsibility                                                    | Key Principle                      |
|:-----------------|:------------------------------------------------------------------|:-----------------------------------|
| **Core**         | Defines contracts and provides default in-memory implementations. | Minimal, foundational dependencies |
| **Libs**         | Provides reusable **internal business logic**.                    | Internal, Business-Oriented        |
| **Integrations** | Interacts with **external infrastructure**.                       | External, Infrastructure-Oriented  |
| **Apps**         | Assembles modules and contains specific business logic.           | Assembly & Specific Logic          |

---

## 2. Key Principles & Notes

* **One-Way Dependencies**: The dependency flow is strictly one-way: `apps` → `libs` / `integrations` → `core`.
* **Stateless Services**: All applications must be designed to be stateless. State is managed by external services via
  the `integrations` layer.
* **Dependency Injection**: The `apps` layer is responsible for assembling all services using a DI container.
* **Testing**: A multi-layered testing strategy (Unit, Integration, E2E, Benchmark) is enforced. Time-based tests must use
  `time-machine`. Performance testing uses `pytest-benchmark`.
* **Observability**: The project is standardized on OpenTelemetry.

---

## 3. Common Commands

These tasks are run from the workspace root using `poethepoet`.

### Initial Setup
The only command needed to set up the development environment.
```bash
# Install all project and development dependencies
uv sync --dev
```

### Daily Workflow

| Purpose                       | Command               |
| :---------------------------- |:----------------------|
| **Run all checks and tests**  | `uv run poe validate` |
| Run all static checks         | `uv run poe check`           |
| Run all tests                 | `uv run poe test`            |
| Format code                   | `uv run poe format`          |

### Performance Testing

| Purpose                       | Command                                                    |
| :---------------------------- |:---------------------------------------------------------- |
| **Run benchmark tests**       | `uv run pytest src/core/tests/benchmark/ --benchmark-only` |
| **Save performance baseline** | `python scripts/run_benchmarks.py --save-baseline`       |
| **Compare with baseline**     | `python scripts/run_benchmarks.py --compare baseline`    |
| **Generate performance report** | `python scripts/run_benchmarks.py --format json`       |

### Advanced Testing
To pass arguments directly to `pytest`, add them after the `poe test` command.
```bash
# Run tests only for the core package
poe test src/core/tests/

# Run tests containing a specific keyword
poe test -k "MyClassName"

# Run benchmark tests with custom settings
uv run pytest src/core/tests/benchmark/ --benchmark-only --benchmark-min-rounds=10

# Run specific benchmark test
uv run pytest src/core/tests/benchmark/test_validators_benchmark.py::TestValidatorsBenchmark::test_env_validator_benchmark --benchmark-only
```
