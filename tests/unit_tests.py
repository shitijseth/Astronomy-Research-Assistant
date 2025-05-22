import pkg_resources
import requests
from astronomy_mcp_server import get_planet_dynamic

def test_urllib3_version():
    """urllib3 must be < 2 until OpenSSL ≥ 1.1.1 is guaranteed."""
    v = pkg_resources.get_distribution("urllib3").parsed_version
    assert v < pkg_resources.parse_version("2"), "urllib3 version too new"

def test_planet_resource_live_call(monkeypatch):
    """Quick smoke test that the live API returns a radius for Mars."""
    # reduce timeout so CI doesn't hang on network issues
    import astronomy_mcp_server as srv
    monkeypatch.setattr(srv, "SOLAR_SYSTEM_API",
                        "https://api.le-systeme-solaire.net/rest/bodies/{id}")
    text = get_planet_dynamic("Mars")
    assert "radius" in text.lower() or "Mean radius" in text
