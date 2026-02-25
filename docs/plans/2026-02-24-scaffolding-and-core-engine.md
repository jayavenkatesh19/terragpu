# Scaffolding & Core Engine Design

**Date:** 2026-02-24
**Covers:** Issue #7 (Scaffolding), Issue #1 (Core Engine)
**Status:** Approved

---

## Decisions

- **License:** MIT
- **Python:** 3.10+
- **Build backend:** hatchling (PEP 621 native)

---

## Issue #7 — Scaffolding

### Package Layout

Exactly as specified in the v1 design doc Section 12. All subpackages created with `__init__.py` stubs.

### pyproject.toml

Core deps (CPU-safe): xarray, dask[complete], rasterio, pystac-client, stackstac

Optional dependency groups:
- `[gpu]`: cupy, cuml, cudf, dask-cuda
- `[torch]`: torch, torchgeo
- `[viz]`: datashader, holoviews
- `[dev]`: ruff, mypy, pytest, pytest-cov, pre-commit

### CI (GitHub Actions)

- Lint + type check job (CPU-only, every PR)
- Unit tests with CPU fallback (Python 3.10 + 3.12 matrix)
- No GPU CI in v1

### Pre-commit

- ruff (lint + format)
- mypy

### Other Files

- MIT LICENSE
- .gitignore (Python)
- CONTRIBUTING.md
- Updated README with project vision and install instructions

---

## Issue #1 — Core Engine

### context.py

- Frozen dataclass with fields: crs, resolution, nodata_policy, chunks, device
- `tg.context()` factory with sensible defaults (crs=None, resolution=None, chunks="auto", device="auto")
- `.with_overrides(**kwargs)` returns new Context (immutable)
- Device auto-detection: check CUDA availability, fall back to CPU

### rasterstack.py

- Wraps `xr.DataArray` (Dask-backed) + `Context`
- Immutable: every fluent method returns a new RasterStack
- Fluent methods (stub signatures, raise NotImplementedError for unimplemented epics):
  - `.cloud_mask()`, `.compute_index()`, `.temporal_composite()`
  - `.resample()`, `.reproject()`, `.select_bands()`, `.apply()`
- Terminal methods: `.compute()`, `.export()`, `.plot()`
- Interop: `.to_torch()`, `.to_xarray()`, `.to_numpy()`, `.to_cupy()`, `.to_dataframe()`
- Properties: `.crs`, `.resolution`, `.bounds`, `.bands`, `.timestamps`, `.shape`
- `__repr__` with shape, bands, CRS, lazy status

### __init__.py

- Expose `tg.context()`, `__version__`

### Tests

- `test_context.py`: creation, defaults, with_overrides, device detection
- `test_rasterstack.py`: construction, metadata, immutability, compute()
