import terragpu as tg


def test_version():
    assert hasattr(tg, "__version__")


def test_context_accessible():
    ctx = tg.context()
    assert ctx is not None
    assert ctx.device in ("gpu", "cpu")


def test_top_level_api_surface():
    """Verify the public API is importable."""
    assert callable(tg.context)
    assert hasattr(tg, "RasterStack")
    assert hasattr(tg, "Context")
