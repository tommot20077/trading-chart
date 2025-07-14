# Development Workflows & Cheatsheet

This document provides a quick reference for common development tasks in this monorepo.

**Golden Rule**: All commands should be executed from the workspace root.

---

## 1. Initial Setup

This section is for new users cloning the project for the first time.

### For Developers (Full Workspace)
If you are a developer working on this project, this is the only command you need after cloning. It will install all project packages and all development tools.

```bash
# 1. Clone the repository and enter it
git clone <repo_url> && cd <project_name>

# 2. Create a virtual environment and install everything
uv venv
uv sync --dev
```

### For Users (Using a single package)
If you only want to use a specific package from this repository as a library in your own project.

```bash
# Install only the 'core' library and its production dependencies
uv pip install <path_to_trading_chart>/src/core
```

---

## 2. Common Tasks (via `poethepoet`)

These tasks are defined in the root `pyproject.toml` under `[tool.poe.tasks]`. You run them with `poe <task_name>`.

| Purpose                       | Command          |
| :---------------------------- | :--------------- |
| **Run all checks and tests**  | `poe validate`   |
| Run all static checks         | `poe check`      |
| Format code                   | `poe format`     |
| Check for lint errors         | `poe lint`       |
| Run type checking             | `poe check-types`|
| Run all tests                 | `poe test`       |
| Run tests with coverage       | `poe test-cov`   |

---

## 3. Advanced Testing

To pass arguments directly to `pytest`, add them after the `poe test` command.

| Scenario                               | Command                                                              |
| :------------------------------------- | :------------------------------------------------------------------- |
| **Run tests by directory**             | `poe test src/core/tests/`                                           |
| **Run tests in a specific file**       | `poe test src/libs/auth/tests/unit/test_services.py`                 |
| **Run tests by keyword**               | `poe test -k "Repository and not InMemory"`                          |
| **Run a specific test class or method**| `poe test src/core/tests/unit/models/base/test_exception.py::TestCoreError` |
| **Run tests by marker**                | `poe test -m "slow"`                                                 |

---

## 4. Benchmark Testing

Performance testing using `pytest-benchmark` for accurate measurements.

### Quick Benchmark Commands

| Purpose                          | Command                                                    |
| :------------------------------- | :--------------------------------------------------------- |
| **Run all benchmarks**          | `uv run pytest src/core/tests/benchmark/ --benchmark-only` |
| **Run specific benchmark**      | `uv run pytest src/core/tests/benchmark/test_validators_benchmark.py::TestValidatorsBenchmark::test_env_validator_benchmark --benchmark-only` |
| **Custom benchmark settings**   | `uv run pytest src/core/tests/benchmark/ --benchmark-only --benchmark-min-rounds=10 --benchmark-max-time=5.0` |

### Benchmark Scripts

| Purpose                          | Command                                                    |
| :------------------------------- | :--------------------------------------------------------- |
| **Basic benchmark run**         | `python scripts/run_benchmarks.py`                        |
| **Save performance baseline**   | `python scripts/run_benchmarks.py --save-baseline`       |
| **Compare against baseline**    | `python scripts/run_benchmarks.py --compare baseline`    |
| **Generate JSON report**        | `python scripts/run_benchmarks.py --format json`         |
| **Generate histogram**          | `python scripts/run_benchmarks.py --format histogram`    |

### Advanced Benchmark Analysis

| Purpose                          | Command                                                    |
| :------------------------------- | :--------------------------------------------------------- |
| **Save custom baseline**        | `uv run pytest src/core/tests/benchmark/ --benchmark-only --benchmark-save=my_baseline` |
| **Compare with baseline**       | `uv run pytest src/core/tests/benchmark/ --benchmark-only --benchmark-compare=my_baseline` |
| **Export to JSON**              | `uv run pytest src/core/tests/benchmark/ --benchmark-only --benchmark-json=results.json` |
| **Generate histogram**          | `uv run pytest src/core/tests/benchmark/ --benchmark-only --benchmark-histogram=histogram` |

---

## 5. Dependency Management

Dependencies are managed per-package using the `--package` flag.

| Scenario                               | Command                                                     |
| :------------------------------------- | :---------------------------------------------------------- |
| Add a dependency to `core`             | `uv pip install --package trading-chart-core "pydantic"`    |
| Add a dependency to the `auth` library | `uv pip install --package trading-chart-auth "pyjwt"`       |
| Add a dev dependency to the workspace  | `uv pip install --dev "pytest-mock"`                        |
| Remove a dependency from `core`        | `uv pip uninstall --package trading-chart-core "pydantic"`  |

---
