"""
URI reasoning utilities for Measurement Set v4.

Provides heuristic inference of the MSv4 backend type from a URI string,
using the Python standard library only. Stubs are provided for checks that
require network access or third-party libraries.
"""

from __future__ import annotations

import re
from os.path import join as pjoin, isfile
from enum import Enum
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


# CASA Table magic number: first 4 bytes of table.dat
_CASA_MAGIC = b"\xbe\xbe\xbe\xbe"

# Matches the RDB file header "REDIS" followed by a 4-digit version number,
# e.g. b"REDIS0006". Group 1 captures the version as bytes.
_REDIS_MAGIC_RE = re.compile(b"REDIS(\d{4})")

# Files indicating the presence of ZARR Datasets
_ZARR_VERSION_MARKERS = {".zattrs": 2, "zarr.json": 3}

class MSv4Backend(Enum):
    """Known MSv4 storage backends."""

    CASA_TABLE = "casa_table"
    """CASA Measurement Set v2 stored in a CASA table directory."""

    ZARR = "zarr"
    """Zarr-backed MSv4 store (local or remote object store)."""

    MEERKAT = "meerkat"
    """MeerKAT archive accessible via xarray-kat."""

    UNKNOWN = "unknown"
    """Backend could not be determined."""


def infer_backend(uri: str, *, strict: bool = False) -> MSv4Backend:
    """Heuristically infer the MSv4 backend from *uri*.

    Detection proceeds from most to least specific:

    1. :attr:`MSv4Backend.CASA_TABLE` — local directory whose ``table.dat``
       starts with the ``BEBEBEBE`` magic bytes (always checked).
    2. :attr:`MSv4Backend.MEERKAT` — ``http``/``https`` URI with
       ``kat.ac.za`` in the netloc and an ``.rdb`` path extension.
       In strict mode the REDIS magic number is also verified via a
       partial GET; see :func:`check_rdb_magic`.
    3. :attr:`MSv4Backend.ZARR` — local directory containing ``.zattrs``
       (Zarr v2) or ``zarr.json`` (Zarr v3); or a remote ``http``/``https``
       URI for which :func:`check_remote_zarr` returns ``True``.
       For other remote schemes (``s3``, ``gs``, …) Zarr is *assumed* in
       non-strict mode; in strict mode a :exc:`NotImplementedError` is raised.
    4. :attr:`MSv4Backend.UNKNOWN` — none of the above matched.

    Parameters
    ----------
    uri:
        A file path or URL string identifying the data store.
    strict:
        When ``True``, confirm backend identity via I/O in addition to URI
        heuristics: verify the REDIS magic for MeerKAT URIs, and raise
        :exc:`NotImplementedError` for remote schemes that cannot be probed
        with the standard library.

    Returns
    -------
    MSv4Backend
        The inferred backend type.
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()

    # --- CASA Table (always checked — local and cheap) ---
    if _is_local(scheme) and _is_casa_table(uri):
        return MSv4Backend.CASA_TABLE

    # --- MeerKAT ---
    if (
        scheme in ("http", "https")
        and "kat.ac.za" in parsed.netloc
        and parsed.path.endswith(".rdb")
    ):
        if strict and not check_rdb_magic(uri):
            return MSv4Backend.UNKNOWN
        return MSv4Backend.MEERKAT

    # --- Zarr ---

    # Always perform a strict check on a local URI
    if _is_local(scheme):
        return MSv4Backend.ZARR if _is_local_zarr(uri) else MSv4Backend.UNKNOWN

    if strict:
        if scheme in ("http", "https"):
            return MSv4Backend.ZARR if check_remote_zarr(uri) else MSv4Backend.UNKNOWN

        # Unknown remote scheme (s3, gs, ...): assume Zarr in non-strict mode
        raise NotImplementedError(
            f"infer_backend with strict=True is not implemented for scheme {scheme!r}; "
            "provide an implementation for this object store."
        )
    else:
        # If we're not performing strict checks then assume
        # the most general case: a Zarr Dataset
        return MSv4Backend.ZARR


def check_rdb_magic(uri: str) -> bool:
    """Check whether *uri* points to an RDB file by verifying the REDIS magic.

    Issues a partial GET request (``Range: bytes=0-4``) and compares the
    response body against the REDIS magic bytes ``b"REDIS"``.

    Only ``http`` and ``https`` schemes are supported. Other schemes raise
    :exc:`NotImplementedError`.

    Parameters
    ----------
    uri:
        An ``http`` or ``https`` URL ending with ``.rdb``.

    Returns
    -------
    bool
        ``True`` if the file begins with the REDIS magic bytes,
        ``False`` if the request fails or the magic does not match.

    Raises
    ------
    NotImplementedError
        If the URI scheme is not ``http`` or ``https``.
    """
    scheme = urlparse(uri).scheme.lower()
    if scheme not in ("http", "https"):
        raise NotImplementedError(
            f"check_rdb_magic is not implemented for scheme {scheme!r}; "
            "provide an implementation for this backend."
        )
    req = Request(uri, headers={"Range": "bytes=0-8"})
    try:
        with urlopen(req, timeout=10) as resp:
            if m := _REDIS_MAGIC_RE.match(resp.read(9)):
                return 6 <= int(m.group(1)) <= 10
            return False
    except (HTTPError, URLError):
        return False


def check_remote_zarr(uri: str) -> bool:
    """Check whether a remote *uri* refers to a Zarr store.

    Probes for ``.zattrs`` (Zarr v2) or ``zarr.json`` (Zarr v3) at the URI
    prefix using HTTP HEAD requests.

    Only ``http`` and ``https`` schemes are supported. Other schemes raise
    :exc:`NotImplementedError`.

    Parameters
    ----------
    uri:
        A remote ``http`` or ``https`` URL.

    Returns
    -------
    bool
        ``True`` if either marker is found, ``False`` otherwise (including
        on connection errors).

    Raises
    ------
    NotImplementedError
        If the URI scheme is not ``http`` or ``https``.
    """
    scheme = urlparse(uri).scheme.lower()
    if scheme not in ("http", "https"):
        raise NotImplementedError(
            f"check_remote_zarr is not implemented for scheme {scheme!r}; "
            "provide an implementation for this object store."
        )
    base = uri.rstrip("/")
    for marker in set(_ZARR_VERSION_MARKERS.keys()):
        req = Request(f"{base}/{marker}", method="HEAD")
        try:
            with urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return True
        except (HTTPError, URLError):
            pass
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_local(scheme: str) -> bool:
    """Return True for file-system paths (empty scheme or 'file')."""
    return scheme in ("", "file")


def _is_casa_table(path: str) -> bool:
    """Return True if *path* is a CASA table directory."""
    table_dat = pjoin(path, "table.dat")
    if not isfile(table_dat):
        return False
    try:
        with open(table_dat, "rb") as fh:
            return fh.read(4) == _CASA_MAGIC
    except OSError:
        return False


def _is_local_zarr(path: str) -> bool:
    """Return True if *path* is a local Zarr store directory."""
    return any(isfile(pjoin(path, m)) for m in _ZARR_VERSION_MARKERS.keys())
