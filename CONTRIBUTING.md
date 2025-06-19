# Contributing

## Introduction

First off, thank you for considering contributions to RedisVL! We value community contributions and appreciate your interest in helping make this project better.

## Types of Contributions We Need

You may already know what you want to contribute -- a fix for a bug you encountered, or a new feature your team wants to use.

If you don't know what to contribute, keep an open mind! Here are some valuable ways to contribute:

- **Bug fixes**: Help us identify and resolve issues
- **Feature development**: Add new functionality that benefits the community
- **Documentation improvements**: Enhance clarity, add examples, or fix typos
- **Bug triaging**: Help categorize and prioritize issues
- **Writing tutorials**: Create guides that help others use RedisVL
- **Testing**: Write tests or help improve test coverage

## Getting Started

Here's how to get started with your code contribution:

1. **Fork the repository**: Create your own fork of this repo
2. **Set up your development environment**: Follow the setup instructions below
3. **Make your changes**: Apply the changes in your forked codebase
4. **Test your changes**: Ensure your changes work and don't break existing functionality
5. **Submit a pull request**: If you like the change and think the project could use it, send us a pull request

## Development Environment Setup

### Prerequisites

- **Python**: RedisVL supports Python 3.8 and above
- **Docker**: Required for running Redis Stack and integration tests
- **UV**: Modern Python package manager for fast dependency management

### Installing UV

RedisVL uses [UV](https://docs.astral.sh/uv/) for fast, modern Python dependency management. Choose your preferred installation method:

#### Standalone Installer (Recommended)

**macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Alternative Installation Methods

**Homebrew (macOS):**
```bash
brew install uv
```

**pipx:**
```bash
pipx install uv
```

**pip:**
```bash
pip install uv
```

For more installation options, see the [official UV installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Project Setup

Once UV is installed, set up the project dependencies:

```bash
# Install all dependencies including development and optional extras
uv sync --all-extras
```

This will create a virtual environment and install all necessary dependencies for development.

## Using the Makefile

We provide a comprehensive Makefile to streamline common development tasks. Here are the available commands:

| Command | Description |
|---------|-------------|
| `make install` | Installs all dependencies using UV |
| `make redis-start` | Starts Redis Stack in a Docker container on ports 6379 and 8001 |
| `make redis-stop` | Stops the Redis Stack Docker container |
| `make format` | Runs code formatting and import sorting |
| `make check-types` | Runs mypy type checking |
| `make lint` | Runs formatting, import sorting, and type checking |
| `make test` | Runs tests, excluding those that require API keys and/or remote network calls |
| `make test-all` | Runs all tests, including those that require API keys and/or remote network calls |
| `make test-notebooks` | Runs all notebook tests |
| `make check` | Runs all linting targets and a subset of tests |
| `make docs-build` | Builds the documentation |
| `make docs-serve` | Serves the documentation locally |
| `make clean` | Removes all generated files (cache, coverage, build artifacts, etc.) |

**Quick Start Example:**
```bash
# Set up the project
make install

# Start Redis Stack
make redis-start

# Run linting and tests
make check

# Stop Redis when done
make redis-stop
```

## Code Quality and Testing

### Linting and Formatting

We maintain high code quality standards. Before submitting your changes, ensure they pass our quality checks:

```bash
# Format code and sort imports
make format

# Check types
make check-types

# Or run all linting checks at once
make lint
```

You can also run these commands directly with UV:
```bash
uv run ruff format
uv run ruff check --fix
uv run mypy redisvl
```

### Running Tests

#### TestContainers

RedisVL uses [Testcontainers Python](https://testcontainers-python.readthedocs.io/) for integration tests. Testcontainers provisions throwaway, on-demand containers for development and testing.

**Requirements:**
- Local Docker installation such as:
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - [Docker Engine on Linux](https://docs.docker.com/engine/install/)

#### Test Commands

```bash
# Run tests without external API calls
make test

# Run all tests including those requiring API keys
make test-all

# Run tests on a specific file
uv run pytest tests/unit/test_fields.py -v

# Run tests with coverage
uv run pytest --cov=redisvl --cov-report=html
```

**Note:** Tests requiring external APIs need appropriate API keys set as environment variables.

## Documentation

Documentation is served from the `docs/` directory and built using Sphinx.

### Building and Serving Docs

```bash
# Build the documentation
make docs-build

# Serve documentation locally at http://localhost:8000
make docs-serve
```

Or using UV directly:
```bash
# Build docs
uv run sphinx-build -b html docs docs/_build/html

# Serve docs
uv run python -m http.server 8000 --directory docs/_build/html
```

## Redis Setup

To develop and test RedisVL applications, you need Redis with Search & Query features. You have several options:

### Option 1: Redis Stack with Docker (Recommended for Development)

```bash
# Start Redis Stack with RedisInsight GUI
make redis-start

# This runs:
# docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Stop when finished
make redis-stop
```

This also provides the [FREE RedisInsight GUI](https://redis.io/insight/) at `http://localhost:8001`.

### Option 2: Redis Cloud

For production-like testing, use [Redis Cloud](https://redis.io/cloud/) which provides managed Redis instances with Search & Query capabilities.

## Reporting Issues

### Security Vulnerabilities

**⚠️ IMPORTANT**: If you find a security vulnerability, do NOT open a public issue. Email [Redis OSS](mailto:oss@redis.com) instead.

**Questions to determine if it's a security issue:**
- Can I access something that's not mine, or something I shouldn't have access to?
- Can I disable something for other people?

If you answer *yes* to either question, it's likely a security issue.

### Bug Reports

When filing a bug report, please include:

1. **Python version**: What version of Python are you using?
2. **Package versions**: What versions of `redis` and `redisvl` are you using?
3. **Steps to reproduce**: What did you do?
4. **Expected behavior**: What did you expect to see?
5. **Actual behavior**: What did you see instead?
6. **Environment**: Operating system, Docker version (if applicable)
7. **Code sample**: Minimal code that reproduces the issue

## Suggesting Features

Before suggesting a new feature:

1. **Check existing issues**: Search our [issue list](https://github.com/redis/redis-vl-python/issues) to see if someone has already proposed it
2. **Consider the scope**: Ensure the feature aligns with RedisVL's goals
3. **Provide details**: If you don't see anything similar, open a new issue that describes:
   - The feature you would like
   - How it should work
   - Why it would be beneficial
   - Any implementation ideas you have

## Pull Request Process

1. **Fork and create a branch**: Create a descriptive branch name (e.g., `fix-search-bug` or `add-vector-similarity`)
2. **Make your changes**: Follow our coding standards and include tests
3. **Test thoroughly**: Ensure your changes work and don't break existing functionality
4. **Update documentation**: Add or update documentation as needed
5. **Submit your PR**: Include a clear description of what your changes do

### Review Process

- The core team reviews Pull Requests regularly
- We provide feedback as soon as possible
- After feedback, we expect a response within two weeks
- PRs may be closed if they show no activity after this period

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows our style guidelines (`make lint` passes)
- [ ] Tests pass (`make test` passes)
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains what changes were made and why

## Getting Help

If you need help or have questions:

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check our [documentation](https://www.redisvl.com/) for guides and examples

Thank you for contributing to RedisVL! 🚀
