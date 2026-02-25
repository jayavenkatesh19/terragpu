"""TerraGPU — GPU-accelerated geospatial raster processing."""

__version__ = "0.1.0"

from terragpu.context import Context, context
from terragpu.rasterstack import RasterStack

__all__ = ["Context", "RasterStack", "context", "__version__"]
