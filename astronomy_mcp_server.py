# ============================
# astronomy_mcp_server.py
# Stage 1 prototype MCP server for Astronomy Research Assistant
# Requires:  Python >=3.10, FastMCP (pip install "mcp[cli]"), requests, astroquery, pandas
# Optional but recommended: python-dotenv to load NASA_API_KEY from .env
# ----------------------------
from __future__ import annotations
import os
import textwrap
from typing import Optional

import requests
import mcp
from mcp.server.fastmcp import FastMCP

try:
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
except ImportError:
    NasaExoplanetArchive = None  # astroquery is optional: pip install astroquery

# ---------------------------------------------------------------------------
# 1 — initialise MCP server
# ---------------------------------------------------------------------------
# IMPORTANT:  the global variable must be named one of {mcp, server, app}
# so that `mcp run …` or `mcp dev …` can auto‑discover it.

mcp = FastMCP("AstronomyAssistant")

# ---------------------------------------------------------------------------
# 2 — helper:  read your NASA API key from env (var or .env file)
# ---------------------------------------------------------------------------
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")

# ---------------------------------------------------------------------------
# 3 — TOOL: get Astronomy Picture of the Day (APOD)
# ---------------------------------------------------------------------------
@mcp.tool()
def get_apod(date: Optional[str] = None) -> str:
    """Return NASA Astronomy Picture of the Day (APOD) title + explanation.

    Parameters
    ----------
    date : str, optional
        Date in YYYY-MM-DD format. If omitted, today is used.
    """
    params = {"api_key": NASA_API_KEY}
    if date:
        params["date"] = date

    resp = requests.get("https://api.nasa.gov/planetary/apod", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    title = data.get("title", "Astronomy Picture of the Day")
    explanation = data.get("explanation", "")
    return f"{title}:\n{textwrap.shorten(explanation, width=600)}"

# ---------------------------------------------------------------------------
# 4 — TOOL: basic Exoplanet Archive query by planet name
# ---------------------------------------------------------------------------
@mcp.tool()
def get_exoplanet_info(planet_name: str) -> str:
    """Return a short summary (radius, orbital period, discovery year) for a given exoplanet.

    `planet_name` should match the official name used by the NASA Exoplanet Archive.
    """
    # Prefer astroquery if available (more convenient) otherwise fallback to TAP REST API
    if NasaExoplanetArchive is not None:
        try:
            table = NasaExoplanetArchive.query_planet(f"pl_name=\"{planet_name}\"")
            if not table:
                return f"No exoplanet named '{planet_name}' found in archive."
            row = table[0]
            radius = row.get("pl_rade", "?")
            period = row.get("pl_orbper", "?")
            discyr = row.get("disc_year", "?")
        except Exception as e:
            return f"Archive query failed: {e}"
    else:
        query = (
            "select pl_name,pl_rade,pl_orbper,disc_year "
            "from ps "
            f"where pl_name='" + planet_name.replace("'", "''") + "'"
        )
        params = {"query": query, "format": "json"}
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return f"Archive request failed with status {resp.status_code}"
        data = resp.json()
        if not data:
            return f"No exoplanet named '{planet_name}' found."
        record = data[0]
        radius = record.get("pl_rade", "?")
        period = record.get("pl_orbper", "?")
        discyr = record.get("disc_year", "?")
    return (
        f"Exoplanet {planet_name}: radius ≈ {radius} Earth‑radii, orbital period ≈ {period} days, "
        f"discovered in {discyr}."
    )

# ---------------------------------------------------------------------------
# 5 — RESOURCE:  provide planet facts via URI       e.g. planet://Mars
# ---------------------------------------------------------------------------
_planetary_data = {
    "Mars": {"distance_from_sun_au": 1.52, "radius_km": 3389.5},
    "Earth": {"distance_from_sun_au": 1.0, "radius_km": 6371},
}

@mcp.resource("planet://{name}")
def planet_resource(name: str) -> str:
    """Return basic facts for a Solar‑System planet (local demo data)."""
    info = _planetary_data.get(name.capitalize())
    if not info:
        raise ValueError(f"No data for planet '{name}'.")
    return (
        f"{name.capitalize()} is {info['distance_from_sun_au']} AU from the Sun "
        f"with a mean radius of {info['radius_km']} km."
    )
