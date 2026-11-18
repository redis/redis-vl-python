from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("redisvl")
except PackageNotFoundError:
    # Package is not installed (e.g., during development)
    __version__ = "0.0.0.dev"
