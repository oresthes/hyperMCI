# Contributing to hyperMCI

Thank you for your interest in contributing! Contributions of all kinds are welcome — bug reports, documentation improvements, new features, and more.

## Getting Started

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) for dependency management
- [pre-commit](https://pre-commit.com/) for code quality hooks (optional but recommended)

### Setting Up a Development Environment

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/hyperMCI.git
cd hyperMCI

# 2. Install all dependencies (including dev dependencies)
poetry install --with dev

# 3. (Optional) Install pre-commit hooks
poetry run pre-commit install
```

## Running Tests

```bash
poetry run pytest
```

To run with coverage:

```bash
poetry run pytest --cov=hyperMCI --cov-report=term-missing
```

## Code Style

This project uses:

- **black** for code formatting
- **ruff** for linting
- **mypy** for static type checking

Run all checks at once:

```bash
poetry run black .
poetry run ruff check .
poetry run mypy hyperMCI
```

Or use pre-commit to run them automatically before each commit:

```bash
poetry run pre-commit run --all-files
```

## Submitting a Pull Request

1. **Open an issue first** for anything beyond small fixes, so we can discuss the approach before you invest time writing code.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. Make your changes, add tests, and ensure all checks pass.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Push and open a pull request against `main`.

## Reporting Bugs

Please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Your Python version and OS
- The minimal code to reproduce the issue
- The full error traceback

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to abide by its terms.
