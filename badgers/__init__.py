"""Top-level package for Badgers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("badgers")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

