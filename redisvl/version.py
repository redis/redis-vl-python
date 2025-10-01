try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("redisvl")
except PackageNotFoundError:
    # Package is not installed (e.g., during development)
    __version__ = "0.0.0.dev"
