"""
Sanity check that all modules can be imported.

This is a simple script that can be run from the root of the project
to check that all modules can be imported. It is useful for catching
import errors that may not be caught by the test suite. It is recommended
to be run without optional dependencies installed.

Usage:
    uv run python -m tests.test_imports redisvl
"""

import importlib
import pkgutil
import sys
import traceback
from typing import Iterable


def iter_modules(package_name: str) -> Iterable[str]:
    """Iterate over all modules in a package, including subpackages."""
    pkg = importlib.import_module(package_name)
    yield package_name

    if hasattr(pkg, "__path__"):
        for module_info in pkgutil.walk_packages(
            pkg.__path__, prefix=pkg.__name__ + "."
        ):
            yield module_info.name


def sanity_check_imports(package_name: str) -> int:
    """Check that all modules and submodules in a package can be imported."""
    failures = []

    for fullname in iter_modules(package_name):
        try:
            importlib.import_module(fullname)
            print(f"[ OK ] {fullname}")
        except Exception as e:
            print(f"[FAIL] {fullname}: {e.__class__.__name__}: {e}")
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            failures.append((fullname, tb_str))

    print("\n=== Summary ===")
    if not failures:
        print("All modules imported successfully.")
        return 0
    else:
        print(f"{len(failures)} module(s) failed to import:\n")
        for name, tb in failures:
            print(f"--- {name} ---")
            print(tb)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <package_name>")
        sys.exit(2)

    pkg_name = sys.argv[1]
    sys.exit(sanity_check_imports(pkg_name))
