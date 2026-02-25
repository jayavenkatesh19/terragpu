import dask.array as da
import numpy as np
import pytest
import xarray as xr

from terragpu.context import context
from terragpu.rasterstack import RasterStack


@pytest.fixture
def sample_stack():
    """Create a small Dask-backed xarray DataArray for testing."""
    ctx = context(crs="EPSG:32637", resolution=10, device="cpu")
    np_data = np.random.rand(3, 100, 100).astype(np.float32)
    data = da.from_array(np_data, chunks=(3, 50, 50))
    coords = {
        "band": ["B04", "B03", "B02"],
        "y": np.linspace(0, 990, 100),
        "x": np.linspace(0, 990, 100),
    }
    arr = xr.DataArray(data, dims=["band", "y", "x"], coords=coords)
    return RasterStack(arr, ctx)


class TestRasterStackProperties:
    def test_bands(self, sample_stack):
        assert list(sample_stack.bands) == ["B04", "B03", "B02"]

    def test_shape(self, sample_stack):
        assert sample_stack.shape == (3, 100, 100)

    def test_crs(self, sample_stack):
        assert sample_stack.crs == "EPSG:32637"

    def test_resolution(self, sample_stack):
        assert sample_stack.resolution == 10


class TestRasterStackCompute:
    def test_compute_returns_numpy(self, sample_stack):
        result = sample_stack.compute()
        assert isinstance(result, xr.DataArray)
        assert not isinstance(result.data, da.Array)  # materialized, not dask

    def test_to_numpy(self, sample_stack):
        arr = sample_stack.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 100, 100)


class TestRasterStackImmutability:
    def test_select_bands_returns_new_stack(self, sample_stack):
        new_stack = sample_stack.select_bands(["B04"])
        assert new_stack is not sample_stack
        assert list(new_stack.bands) == ["B04"]
        assert list(sample_stack.bands) == ["B04", "B03", "B02"]


class TestRasterStackRepr:
    def test_repr_contains_key_info(self, sample_stack):
        r = repr(sample_stack)
        assert "RasterStack" in r
        assert "EPSG:32637" in r
        assert "3" in r  # band count
