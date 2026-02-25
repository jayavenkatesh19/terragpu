# Contributing to TerraGPU

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b feat/your-feature`
4. Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

- Write tests first, then implementation
- Run tests: `pytest -v`
- Lint: `ruff check terragpu/ tests/`
- Format: `ruff format terragpu/ tests/`
- Type check: `mypy terragpu/`

## Pull Requests

- Keep PRs focused on a single change
- Ensure all tests pass and linting is clean
- Write a clear description of what changed and why
