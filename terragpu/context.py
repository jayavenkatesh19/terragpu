"""Workspace configuration for TerraGPU processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


def _detect_device() -> str:
    """Check for CUDA availability, return 'gpu' or 'cpu'."""
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return "gpu"
    except Exception:
        return "cpu"


@dataclass(frozen=True)
class Context:
    """Workspace configuration for a TerraGPU processing pipeline."""

    crs: str | None = None
    resolution: float | None = None
    nodata_policy: str = "mask"
    chunks: str | dict[str, Any] | None = "auto"
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device == "auto":
            object.__setattr__(self, "device", _detect_device())

    def with_overrides(self, **kwargs: Any) -> Context:
        """Return a new Context with specified fields overridden."""
        valid_fields = {f.name for f in fields(self)}
        for key in kwargs:
            if key not in valid_fields:
                raise TypeError(f"Unknown field: {key}")
        return replace(self, **kwargs)


def context(**kwargs: Any) -> Context:
    """Create a TerraGPU context with sensible defaults."""
    return Context(**kwargs)
