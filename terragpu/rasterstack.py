"""Fluent API for GPU-accelerated raster processing."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

from terragpu.context import Context


class RasterStack:
    """Fluent API for GPU-accelerated raster processing.

    Wraps a Dask-backed xarray DataArray. All operations return a new
    RasterStack (immutable). Nothing executes until a terminal method
    (.compute(), .export(), .to_*()) is called.
    """

    def __init__(self, data: xr.DataArray, ctx: Context) -> None:
        self._data = data
        self._ctx = ctx

    # -- Metadata properties (no compute needed) --

    @property
    def crs(self) -> str | None:
        return self._ctx.crs

    @property
    def resolution(self) -> float | None:
        return self._ctx.resolution

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        try:
            return (
                float(self._data.x.min()),
                float(self._data.y.min()),
                float(self._data.x.max()),
                float(self._data.y.max()),
            )
        except (AttributeError, ValueError):
            return None

    @property
    def bands(self) -> list[str]:
        if "band" in self._data.dims:
            return list(self._data.band.values)
        return []

    @property
    def timestamps(self) -> list[Any]:
        if "time" in self._data.dims:
            return list(self._data.time.values)
        return []

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    # -- Fluent operations (lazy, return new RasterStack) --

    def select_bands(self, bands: Sequence[str]) -> RasterStack:
        """Select a subset of bands."""
        data = self._data.sel(band=list(bands))
        return RasterStack(data, self._ctx)

    def apply(self, func: Callable[[xr.DataArray], xr.DataArray]) -> RasterStack:
        """Apply a custom function to the underlying data."""
        return RasterStack(func(self._data), self._ctx)

    def cloud_mask(self, **kwargs: Any) -> RasterStack:
        """Apply cloud masking. Requires terragpu.ops.cloud_mask."""
        raise NotImplementedError("Cloud masking not yet implemented (see Issue #4)")

    def compute_index(self, name: str | list[str], **kwargs: Any) -> RasterStack:
        """Compute spectral index/indices. Requires terragpu.indices."""
        raise NotImplementedError("Index computation not yet implemented (see Issue #3)")

    def temporal_composite(
        self, period: str, method: str = "median", **kwargs: Any
    ) -> RasterStack:
        """Temporal compositing. Requires terragpu.ops.composite."""
        raise NotImplementedError("Temporal compositing not yet implemented (see Issue #4)")

    def resample(self, resolution: float, method: str = "bilinear", **kwargs: Any) -> RasterStack:
        """Resample to target resolution. Requires terragpu.ops.resample."""
        raise NotImplementedError("Resampling not yet implemented (see Issue #4)")

    def reproject(self, crs: str, **kwargs: Any) -> RasterStack:
        """Reproject to target CRS. Requires terragpu.ops.resample."""
        raise NotImplementedError("Reprojection not yet implemented (see Issue #4)")

    # -- Terminal operations (trigger Dask execution) --

    def compute(self) -> xr.DataArray:
        """Trigger Dask execution and return materialized DataArray."""
        return self._data.compute()

    def export(self, path: str, format: str = "cog", **kwargs: Any) -> None:
        """Export to file. Requires terragpu.io.export."""
        raise NotImplementedError("Export not yet implemented (see Issue #6)")

    def plot(self, **kwargs: Any) -> Any:
        """Quick visualization. Requires terragpu.viz.plotting."""
        raise NotImplementedError("Plotting not yet implemented")

    # -- Interop (terminal) --

    def to_numpy(self) -> np.ndarray:
        """Compute and return as numpy array."""
        return self._data.compute().values

    def to_xarray(self) -> xr.DataArray:
        """Return the underlying xarray DataArray (still lazy if not computed)."""
        return self._data

    def to_cupy(self) -> Any:
        """Compute and return as cupy array. Requires cupy."""
        try:
            import cupy

            return cupy.asarray(self._data.compute().values)
        except ImportError:
            raise ImportError("cupy is required: pip install terragpu[gpu]") from None

    def to_torch(self, **kwargs: Any) -> Any:
        """Convert to PyTorch tensor. Requires torch."""
        raise NotImplementedError("PyTorch interop not yet implemented (see Issue #5)")

    def to_dataframe(self) -> Any:
        """Convert to tabular format. Requires terragpu.ml.interop."""
        raise NotImplementedError("DataFrame interop not yet implemented (see Issue #5)")

    # -- Repr --

    def __repr__(self) -> str:
        bands = self.bands
        band_str = (
            f"{len(bands)} bands ({', '.join(bands[:3])}{'...' if len(bands) > 3 else ''})"
        )
        shape_str = " x ".join(str(s) for s in self.shape)
        return (
            f"RasterStack({shape_str}, {band_str}, "
            f"crs={self.crs}, resolution={self.resolution})"
        )
