# TerraGPU

GPU-accelerated geospatial raster processing for remote sensing scientists. Declarative pipelines with STAC integration, spectral indices, and PyTorch/TorchGeo interop.

## Vision

TerraGPU provides a scientist-facing API (`Context` + `RasterStack`) on top of Dask-backed xarray. All GPU dependencies are optional extras — the core library installs and runs on CPU-only machines.

## Install

```bash
pip install -e "."

# With GPU support
pip install -e ".[gpu]"

# Development
pip install -e ".[dev]"
```

## Quick Start

```python
import terragpu as tg

ctx = tg.context(crs="EPSG:32637", resolution=10, device="cpu")
print(ctx)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest -v

# Lint and format
ruff check terragpu/ tests/
ruff format terragpu/ tests/

# Type check
mypy terragpu/
```

## License

MIT
