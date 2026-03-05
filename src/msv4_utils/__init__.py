"""msv4-utils: lightweight utilities for Measurement Set v4."""

from msv4_utils.uri import MSv4Backend, infer_backend

__version__ = "0.0.2"

__all__ = [
    # uri
    "MSv4Backend",
    "infer_backend",
]
