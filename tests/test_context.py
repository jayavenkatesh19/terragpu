import pytest

from terragpu.context import Context, context


class TestContextCreation:
    def test_default_context(self):
        ctx = context()
        assert ctx.crs is None
        assert ctx.resolution is None
        assert ctx.nodata_policy == "mask"
        assert ctx.chunks == "auto"
        assert ctx.device in ("gpu", "cpu")

    def test_explicit_params(self):
        ctx = context(crs="EPSG:32637", resolution=10, device="cpu")
        assert ctx.crs == "EPSG:32637"
        assert ctx.resolution == 10
        assert ctx.device == "cpu"

    def test_context_is_immutable(self):
        ctx = context()
        with pytest.raises(AttributeError):
            ctx.crs = "EPSG:4326"


class TestContextOverrides:
    def test_with_overrides_returns_new_context(self):
        ctx = context(crs="EPSG:32637", resolution=10)
        new_ctx = ctx.with_overrides(resolution=20)
        assert new_ctx.resolution == 20
        assert new_ctx.crs == "EPSG:32637"
        assert ctx.resolution == 10  # original unchanged

    def test_with_overrides_rejects_unknown_fields(self):
        ctx = context()
        with pytest.raises(TypeError):
            ctx.with_overrides(nonexistent_field=42)


class TestDeviceDetection:
    def test_auto_device_resolves(self):
        ctx = context(device="auto")
        assert ctx.device in ("gpu", "cpu")

    def test_explicit_cpu(self):
        ctx = context(device="cpu")
        assert ctx.device == "cpu"
