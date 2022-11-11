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

# help: style                          - Sort imports and format with black
.PHONY: style
style: sort-imports format


# help: check-style                    - check code style compliance
.PHONY: check-style
check-style: check-sort-imports check-format


# help: format                         - perform code style format
.PHONY: format
format:
	@black ./redisvl ./tests/


# help: sort-imports                   - apply import sort ordering
.PHONY: sort-imports
sort-imports:
	@isort ./redisvl ./tests/ --profile black


# help: check-lint                     - run static analysis checks
.PHONY: check-lint
check-lint:
	@pylint --rcfile=.pylintrc ./redisvl


# help:
# help: Test
# help: -------

# help: test                           - Run all tests
.PHONY: test
test:
	@python -m pytest

# help: test-verbose                   - Run all tests verbosely
.PHONY: test-verbose
test-verbose:
	@python -m pytest -vv

# help: test-cov                       - Run all tests with coverage
.PHONY: test-cov
test-cov:
	@python -m pytest -vv --cov=./redisvl

# help: cov                            - generate html coverage report
.PHONY: cov
cov:
	@coverage html
	@echo if data was present, coverage report is in ./htmlcov/index.html
