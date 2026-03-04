# msv4-utils

Lightweight utilities for reasoning about Measurement Set v4.
Core functionality depends only on the Python standard library.

## Installation

```bash
pip install msv4-utils
```

## Usage

### Infer the MSv4 backend from a URI

```python
from msv4_utils import infer_backend, MSv4Backend

backend = infer_backend("/path/to/data.ms")
if backend == MSv4Backend.CASA_TABLE:
    # open with xarray-ms
    ...
elif backend == MSv4Backend.ZARR:
    # open with xarray
    ...
elif backend == MSv4Backend.MEERKAT:
    # open with xarray-kat
    ...
```

### MSv4 type constants

```python
from msv4_utils import VISIBILITY, CORRELATED_XDS_TYPES

xds_type = xds.attrs["type"]
if xds_type in CORRELATED_XDS_TYPES:
    print("interferometric data")
```

## Development

```bash
uv sync
uv run pytest
```

### Build docs

```bash
uv run --extra docs sphinx-build docs docs/_build
```
