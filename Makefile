MAKEFLAGS += --no-print-directory

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: Developer Makefile
# help:


SHELL:=/bin/bash

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'


# help:
# help: Style
# help: -------

# help: check                          - run all checks for a commit
.PHONY: check
check: check-format check-sort-imports mypy

# help: format                         - perform code style format
.PHONY: format
format: sort-imports
	@black ./redisvl ./tests/


# help: check-format                   - check code format compliance
.PHONY: check-format
check-format:
	@black --check ./redisvl


# help: sort-imports                   - apply import sort ordering
.PHONY: sort-imports
sort-imports:
	@isort ./redisvl ./tests/ --profile black

# help: check-sort-imports             - check imports are sorted
.PHONY: check-sort-imports
check-sort-imports:
	@isort ./redisvl --check-only --profile black


# help: check-lint                     - run static analysis checks
.PHONY: check-lint
check-lint:
	@pylint --rcfile=.pylintrc ./redisvl

# help: mypy                           - run mypy
.PHONY: mypy
mypy:
	@mypy ./redisvl

# help:
# help: Documentation
# help: -------

# help: docs                           - generate project documentation
.PHONY: docs
docs:
	@cd docs; make html

# help: servedocs                      - Serve project documentation
.PHONY: servedocs
servedocs:
	@cd docs/_build/html/; python -m http.server

# help:
# help: Test
# help: -------

# help: test                           - Run all tests
.PHONY: test
test:
	@python -m pytest --log-level=CRITICAL

# help: test-verbose                   - Run all tests verbosely
.PHONY: test-verbose
test-verbose:
	@python -m pytest -vv -s --log-level=CRITICAL

# help: test-cov                       - Run all tests with coverage
.PHONY: test-cov
test-cov:
	@python -m pytest -vv --cov=./redisvl --log-level=CRITICAL

# help: cov                            - generate html coverage report
.PHONY: cov
cov:
	@coverage html
	@echo if data was present, coverage report is in ./htmlcov/index.html


# help: test-notebooks                 - Run all notebooks
.PHONY: test-notebooks
test-notebooks:
	@cd docs/ && treon -v