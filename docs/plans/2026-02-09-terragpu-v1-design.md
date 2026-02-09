# TerraGPU v1 Design Document

**Date:** 2026-02-09
**Status:** Draft
**Author:** Jaya Venkatesh

---

## 1. Vision

TerraGPU is a Python library that gives remote sensing scientists a declarative, pipeline-style API for processing satellite imagery with automatic GPU acceleration. Scientists think in domain operations — load scenes, mask clouds, compute NDVI, composite over time — and TerraGPU translates that into an optimized GPU execution plan.

### The Problem

Today, a GPU-accelerated remote sensing workflow requires stitching together 8+ libraries: pystac-client for discovery, stackstac for stacking, xarray and dask for lazy arrays, rasterio for I/O, geopandas for vectors, cuDF for GPU dataframes, and cuML for machine learning. Each has its own API, conventions, and failure modes. Scientists spend more time on library plumbing than on science.

**Reference workflow:** [RAPIDS LULC Classification Example](https://docs.rapids.ai/deployment/stable/examples/lulc-classification-gpu/notebook/) — this notebook demonstrates the complexity that TerraGPU aims to eliminate.

### What TerraGPU Replaces

That entire multi-library notebook becomes:

```python
import terragpu as tg

ctx = tg.context(crs="EPSG:32637", resolution=10, chunks="auto")

stack = (
    ctx.load("sentinel-2-l2a", bbox=aoi, date_range="2021-01-01/2021-12-31", max_cloud=30)
    .cloud_mask(method="scl", keep=[3, 4, 5, 6, 11])
    .compute_index(["NDVI", "NDWI"])
    .temporal_composite("annual", method="median")
)

labels = ctx.load("esa-worldcover", bbox=aoi)
model = tg.ml.classify(stack, labels, method="random_forest")
model.predict(stack).export("lulc_2021.tif", format="cog")
```

---

## 2. Core Principles

1. **Scientist-first API** — reads like a remote sensing workflow, not a software engineering exercise
2. **Smart defaults, full manual control** — convention-over-configuration for the 80% case, explicit override for everything via context objects and per-step parameters. No magic that can't be turned off.
3. **GPU by default, CPU as fallback** — automatically uses GPU when available, degrades gracefully to CPU so code is portable
4. **No walled garden** — interop at every stage with PyTorch, xarray, numpy, cupy. You can drop out of TerraGPU at any point
5. **STAC-native** — first-class integration with STAC catalogs as the primary data discovery mechanism
6. **Raster-first** — satellite/aerial imagery is the core domain. Vector support is a future addition.

---

## 3. Target Users

- **Primary:** Remote sensing scientists, earth observation analysts, geospatial data scientists
- **Secondary:** ML engineers building geospatial training pipelines who need fast preprocessing
- **Non-target (v1):** GIS application developers, vector-heavy workflows

---

## 4. Architecture

TerraGPU uses a two-layer architecture: a scientist-facing API layer on top of a backend layer powered by xarray, Dask, and RAPIDS.

```
┌─────────────────────────────────────────────────┐
│                  API Layer                       │
│  tg.context() · RasterStack · tg.ml · tg.indices│
├─────────────────────────────────────────────────┤
│               Backend Layer                      │
│  xarray · Dask · dask-cuda · cupy · cuML ·       │
│  stackstac · rasterio                            │
└─────────────────────────────────────────────────┘
```

**There is no custom DAG engine.** Dask already IS a DAG execution engine — when you perform lazy operations on Dask-backed xarray arrays, Dask builds and optimizes a task graph automatically. TerraGPU's value is the scientist-friendly API on top and GPU backend dispatch underneath.

### 4.1 API Layer (what scientists touch)

#### Context Object

The Context holds workspace-level configuration. It eliminates the need for repetitive parameter passing and keeps the pipeline chain clean.

```python
ctx = tg.context(
    crs="EPSG:32637",        # target CRS (auto-detect from data if omitted)
    resolution=10,            # target resolution in CRS units
    nodata_policy="mask",     # how to handle nodata: "mask", "nan", "zero"
    chunks="auto",            # chunk strategy: "auto", dict, or None
    device="gpu",             # "gpu", "cpu", or "auto"
)
```

- Sensible defaults for everything — `tg.context()` with no arguments is valid
- Any parameter can be overridden per-operation in the pipeline chain
- The context is immutable once created; `.with_overrides()` returns a new context

#### RasterStack

The core fluent object. Internally wraps a Dask-backed xarray DataArray. All operations return a new RasterStack (immutable), enabling method chaining. Lazy by default — Dask builds the computation graph, nothing executes until a terminal operation is called.

```python
class RasterStack:
    def __init__(self, data: xr.DataArray, ctx: Context):
        self._data = data          # Dask-backed xarray DataArray (lazy)
        self._ctx = ctx

    def compute_index(self, name):
        formula = tg.indices.get(name)
        # Adds to the Dask graph — no compute happens yet
        result = formula.apply(self._data)
        return RasterStack(result, self._ctx)

    def compute(self):
        # Triggers Dask DAG execution
        return self._data.compute()
```

**Lazy operations (extend the Dask graph):**
- `.cloud_mask(method, keep, ...)`
- `.compute_index(name_or_list)`
- `.temporal_composite(period, method)`
- `.resample(resolution, method)`
- `.reproject(crs)`
- `.select_bands(bands)`
- `.apply(func)` — custom user function

**Terminal operations (trigger Dask execution):**
- `.compute()` — materialize in GPU/CPU memory
- `.export(path, format, ...)` — write to disk
- `.to_torch()`, `.to_xarray()`, `.to_numpy()`, `.to_cupy()`, `.to_dataframe()`
- `.plot()` — quick visualization

**Metadata access (no compute needed):**
- `.crs`, `.resolution`, `.bounds`, `.bands`, `.timestamps`, `.shape`

#### Dataset Composition

Inspired by TorchGeo, RasterStack supports operator overloading for combining datasets:

```python
# Union: combine bands/sources
combined = sentinel_stack | dem_stack

# Intersection: co-located data (for ML label pairing)
dataset = sentinel_stack & worldcover_labels
```

### 4.2 Backend Layer (I/O, compute, and scheduling)

The backend layer leverages existing battle-tested libraries. Dask handles all task graph construction, optimization, and scheduling. `dask-cuda` provides GPU-aware scheduling.

| Concern | GPU Backend | CPU Fallback |
|---------|------------|--------------|
| Array ops | cupy | numpy |
| Lazy arrays | xarray + Dask | xarray + Dask |
| Tabular ops | cuDF | pandas |
| ML | cuML | scikit-learn |
| I/O | stackstac + rasterio | stackstac + rasterio |
| Scheduling | dask-cuda | dask |

**How lazy execution works:**

1. Each fluent method on RasterStack appends xarray/Dask operations to the underlying Dask graph
2. Dask automatically handles: task graph optimization, memory management, chunk scheduling
3. `dask-cuda` routes array operations to GPU via cupy when a CUDA device is available
4. Terminal operations (`.compute()`, `.export()`, `.to_torch()`) trigger Dask to execute the full graph
5. For data exceeding single-GPU memory, Dask handles chunked execution automatically

**Future optimization (v2+):** Custom Dask graph optimization passes for TerraGPU-specific patterns (e.g., fusing sequential band math into single GPU kernels via `cupy.fuse`, optimizing rechunking order for GPU VRAM).

The backend layer also provides interop bridges:
- **PyTorch:** zero-copy via DLPack (`stack.to_torch()`)
- **xarray:** metadata-preserving conversion (`stack.to_xarray()`)
- **TorchGeo:** GeoDataset-compatible objects for sampler/DataLoader workflows

---

## 5. STAC Integration

STAC is the primary data discovery and ingestion mechanism.

### Discovery

```python
stack = ctx.load(
    "sentinel-2-l2a",                        # STAC collection ID
    bbox=aoi,                                 # GeoJSON or [west, south, east, north]
    date_range="2021-01-01/2021-12-31",       # ISO 8601 range
    max_cloud=30,                             # cloud cover filter (%)
    bands=["B02", "B03", "B04", "B08"],       # optional band selection
    stac_endpoint="https://planetarycomputer.microsoft.com/api/stac/v1"  # default
)
```

### Supported STAC Endpoints (v1)

- Microsoft Planetary Computer (default)
- Element84 Earth Search
- Custom endpoints via `stac_endpoint` parameter

### Sensor Configuration

Sensors are defined as configuration objects that map band names, provide cloud masking strategies, and handle quirks:

```python
# Pre-configured for Sentinel-2 in v1
SENTINEL_2 = SensorConfig(
    collection="sentinel-2-l2a",
    bands={
        "B02": {"name": "blue", "wavelength": 490, "resolution": 10},
        "B03": {"name": "green", "wavelength": 560, "resolution": 10},
        "B04": {"name": "red", "wavelength": 665, "resolution": 10},
        "B08": {"name": "nir", "wavelength": 842, "resolution": 10},
        "B11": {"name": "swir1", "wavelength": 1610, "resolution": 20},
        "B12": {"name": "swir2", "wavelength": 2190, "resolution": 20},
        "SCL": {"name": "scl", "resolution": 20},
    },
    cloud_mask_band="SCL",
    scale_factor=0.0001,
)
```

This configuration-driven approach means adding new sensors (Landsat, Sentinel-1) is a data problem, not a code change.

---

## 6. Spectral Index Registry

### Design Goals

- Ship with common indices pre-registered
- Scientists can add custom indices without touching core code
- Two expression modes: string formulas (simple) and callables (complex)
- All indices compile to GPU-accelerated operations

### String Expressions (simple indices)

```python
tg.indices.register(
    "NDVI",
    formula="(nir - red) / (nir + red)",
    bands={"nir": "B08", "red": "B04"},
    description="Normalized Difference Vegetation Index",
    reference="Rouse et al. 1974"
)

tg.indices.register(
    "EVI",
    formula="2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)",
    bands={"nir": "B08", "red": "B04", "blue": "B02"}
)
```

The expression parser supports:
- Arithmetic operators: `+`, `-`, `*`, `/`, `**`
- Built-in functions: `sqrt()`, `abs()`, `log()`, `clip()`, `where()`
- Numeric constants
- Operator precedence and parentheses
- Compiled to cupy/numpy operations for GPU/CPU execution

### Callable Functions (complex indices)

For indices that require conditionals, matrix operations, or multi-step logic:

```python
@tg.indices.register("TasseledCap_Brightness")
def tc_brightness(bands):
    coeffs = [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872]
    band_keys = ["B02", "B03", "B04", "B08", "B11", "B12"]
    return sum(c * bands[b] for c, b in zip(coeffs, band_keys))
```

### Built-in Indices (v1)

| Index | Formula | Use Case |
|-------|---------|----------|
| NDVI | (NIR - Red) / (NIR + Red) | Vegetation health |
| NDWI | (Green - NIR) / (Green + NIR) | Water detection |
| EVI | 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) | Enhanced vegetation |
| SAVI | 1.5 * (NIR - Red) / (NIR + Red + 0.5) | Soil-adjusted vegetation |
| NDBI | (SWIR1 - NIR) / (SWIR1 + NIR) | Built-up area |
| NBR | (NIR - SWIR2) / (NIR + SWIR2) | Burn severity |

### Band Name Resolution

The registry resolves symbolic band names (e.g., `nir`, `red`) to sensor-specific band IDs (e.g., `B08`, `B04`) using the sensor configuration from the context. This means the same index definition works across sensors.

---

## 7. Raster Operations

### Cloud Masking

```python
# SCL-based (Sentinel-2)
stack.cloud_mask(method="scl", keep=[4, 5, 6, 7, 11])

# Custom mask function
stack.cloud_mask(method="custom", func=lambda scl: scl.isin([4, 5, 6]))

# Generic bitmask (future: Landsat QA_PIXEL)
stack.cloud_mask(method="bitmask", band="QA_PIXEL", bits={1: 0, 3: 0})
```

### Temporal Compositing

```python
stack.temporal_composite("monthly", method="median")
stack.temporal_composite("annual", method="percentile", q=75)
stack.temporal_composite("seasonal", method="mean")
```

Periods: `daily`, `weekly`, `monthly`, `seasonal`, `annual`, or a custom grouping function.
Methods: `median`, `mean`, `max`, `min`, `percentile`, `count` (valid observations).

### Resampling and Reprojection

```python
stack.resample(resolution=20, method="bilinear")
stack.reproject(crs="EPSG:4326")
```

These are handled transparently when the context specifies a target CRS/resolution, but can be called explicitly.

---

## 8. ML Integration

### Boundary

TerraGPU owns **data processing and simple ML**. Deep learning training belongs to TorchGeo/PyTorch.

### cuML Wrappers (`tg.ml`)

```python
# Supervised classification
model = tg.ml.classify(stack, labels, method="random_forest", n_estimators=100)
prediction = model.predict(new_stack)

# Unsupervised clustering
clusters = tg.ml.cluster(stack, method="kmeans", n_clusters=5)

# Available methods (v1)
# classify: random_forest, xgboost
# cluster: kmeans, dbscan
```

Under the hood:
1. Flatten RasterStack spatial dimensions to rows (pixels → tabular)
2. Use cuML for GPU-accelerated training/prediction
3. Reshape predictions back to raster spatial dimensions
4. Return result as RasterStack with geospatial metadata

### PyTorch Interop

```python
# Zero-copy GPU tensor (via DLPack when possible)
tensor = stack.to_torch()
tensor = stack.to_torch(dtype=torch.float32, device="cuda:0")

# Preserves metadata
tensor.geo_metadata  # {"crs": ..., "transform": ..., "bands": ...}
```

### TorchGeo Interop

```python
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader

# Dataset composition (TorchGeo pattern)
dataset = stack & labels  # returns TorchGeo-compatible GeoDataset

# Standard TorchGeo workflow
sampler = RandomGeoSampler(dataset, size=256, length=1000)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)

for batch in dataloader:
    images = batch["image"]  # GPU tensor
    masks = batch["mask"]    # GPU tensor
```

### Other Interop Targets

```python
stack.to_xarray()      # xarray.Dataset with CRS, transform
stack.to_numpy()       # numpy array (CPU)
stack.to_cupy()        # cupy array (GPU)
stack.to_dataframe()   # cuDF DataFrame (GPU) or pandas (CPU)
```

---

## 9. Export & I/O

```python
# GeoTIFF / Cloud-Optimized GeoTIFF
result.export("output.tif", format="cog", compress="deflate")

# Zarr (chunked, good for intermediate results)
result.export("output.zarr", format="zarr")

# NetCDF
result.export("output.nc", format="netcdf")

# GeoParquet (tabular export)
result.export("output.parquet", format="geoparquet")
```

All exports:
- Preserve CRS, transform, nodata values, band names
- Minimize unnecessary GPU→CPU→disk copies
- Show progress bar for large writes
- Overwrite protection by default (`overwrite=False`)

---

## 10. Visualization

TerraGPU does not build custom visualization. Instead, it integrates with existing GPU-native viz tools:

```python
# Quick notebook preview (delegates to datashader/holoviews)
stack.plot()
stack.plot(band="NDVI", cmap="RdYlGn")

# For advanced viz, drop into the ecosystem
import hvplot.xarray
stack.to_xarray().hvplot()
```

---

## 11. v1 Scope

### In Scope
- Context object with full configuration
- RasterStack fluent API with lazy execution (Dask-backed)
- STAC discovery and ingestion (Planetary Computer, Element84)
- Sentinel-2 first-class support (band mappings, SCL cloud masking)
- Extensible spectral index registry (string + callable)
- Core raster ops: cloud mask, temporal composite, resample, reproject
- cuML wrappers: random forest, xgboost, kmeans, dbscan
- PyTorch/TorchGeo interop (to_torch, dataset composition)
- Export: COG, Zarr, NetCDF, GeoParquet
- GPU by default with CPU fallback
- Basic GPU viz integration

### Out of Scope (future versions)
- Custom Dask graph optimization passes (v2 — fusing band math, VRAM-aware rechunking)
- Landsat, Sentinel-1, or other sensor support
- Vector operations (spatial joins, point-in-polygon)
- Deep learning training (TorchGeo's domain)
- Custom STAC server hosting
- Web UI or dashboard
- Distributed multi-node GPU clusters

---

## 12. Package Structure

```
terragpu/
├── __init__.py              # tg.context(), top-level API
├── context.py               # Context object
├── rasterstack.py           # RasterStack fluent API (wraps Dask-backed xarray)
├── stac/
│   ├── discovery.py         # STAC API queries
│   ├── ingestion.py         # COG loading and stacking
│   └── sensors.py           # Sensor configs (Sentinel-2)
├── ops/
│   ├── cloud_mask.py        # Cloud masking operations
│   ├── composite.py         # Temporal compositing
│   ├── resample.py          # Resampling and reprojection
│   └── indices.py           # Index computation (uses registry)
├── indices/
│   ├── registry.py          # Index registry and expression parser
│   └── builtin.py           # Pre-registered indices
├── ml/
│   ├── classify.py          # cuML classification wrappers
│   ├── cluster.py           # cuML clustering wrappers
│   └── interop.py           # PyTorch/TorchGeo bridge
├── io/
│   └── export.py            # Export to COG, Zarr, NetCDF, GeoParquet
└── viz/
    └── plotting.py          # GPU viz tool integration
```

Note: No `engine/` directory. Dask IS the execution engine. The RasterStack translates fluent API calls into Dask-backed xarray operations directly.

---

## 13. Dependencies

### Core
- `cupy` — GPU array operations
- `cudf` — GPU dataframes (for tabular ML path)
- `cuml` — GPU machine learning
- `dask` / `dask-cuda` — lazy computation, task graph, and GPU-aware scheduling
- `pystac-client` — STAC API queries
- `stackstac` — STAC to xarray stacking
- `rasterio` — geospatial I/O (GDAL bindings)
- `xarray` — labeled array metadata layer

### Optional
- `torch` — PyTorch interop (`pip install terragpu[torch]`)
- `torchgeo` — TorchGeo dataset compatibility (`pip install terragpu[torchgeo]`)
- `planetary-computer` — Planetary Computer authentication (`pip install terragpu[pc]`)
- `datashader` / `holoviews` — GPU visualization (`pip install terragpu[viz]`)

### Dev
- `ruff` — linting
- `mypy` — type checking
- `pytest` — testing
- `pre-commit` — git hooks

---

## 14. Open Questions

1. **License:** Apache 2.0 or MIT? Apache 2.0 aligns with RAPIDS ecosystem, MIT is simpler.
2. **Minimum Python version:** 3.10+ (for modern type hints) or 3.9+ (broader compatibility)?
3. **CUDA version requirements:** What minimum CUDA/driver version to target?
4. **Expression parser implementation:** Use an existing library (e.g., `numexpr`, `asteval`) or build a lightweight custom parser?
5. **Zarr version:** Zarr v2 (stable) or Zarr v3 (newer, spec still evolving)?

---

## References

- [RAPIDS LULC Classification Notebook](https://docs.rapids.ai/deployment/stable/examples/lulc-classification-gpu/notebook/)
- [TorchGeo Documentation](https://torchgeo.readthedocs.io/en/stable/)
- [TorchGeo GitHub](https://github.com/torchgeo/torchgeo)
- [STAC Specification](https://stacspec.org/)
- [cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [stackstac Documentation](https://stackstac.readthedocs.io/)
