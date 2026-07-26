import pytest


@pytest.fixture(autouse=True)
def reset_api_globals() -> None:
    from app import api as api_module

    api_module._AUTH_ATTEMPTS.clear()
    api_module._SERVICE_CACHE.clear()
    yield
    api_module._AUTH_ATTEMPTS.clear()
    api_module._SERVICE_CACHE.clear()
