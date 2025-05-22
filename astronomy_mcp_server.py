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

# --------------------------------------------------------------------------- #
# Guard against urllib3 ≥ 2 when OpenSSL is too old
# --------------------------------------------------------------------------- #
try:
    v = pkg_resources.get_distribution("urllib3").parsed_version
    if v >= pkg_resources.parse_version("2"):
        raise RuntimeError(
            "urllib3 >= 2.0 detected but this Python is linked against "
            "OpenSSL < 1.1.1.  Please `pip install \"urllib3<2\"` or use the "
            "provided requirements.txt."
        )
except pkg_resources.DistributionNotFound:
    # urllib3 not yet installed — `pip install -r requirements.txt` will pull the right version
    pass

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
SOLAR_SYSTEM_API = "https://api.le-systeme-solaire.net/rest/bodies/{id}"

def _fetch_planet_json(name: str) -> dict:
    """Return raw JSON for a Solar-System body or raise ValueError if missing."""
    url = SOLAR_SYSTEM_API.format(id=name.lower())
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise ValueError(f"Could not retrieve data for '{name}'. "
                         f"Status code: {r.status_code}")
    data = r.json()
    # The API returns {'isPlanet': False, ...} for junk requests; guard that:
    if data.get("englishName", "").lower() != name.lower():
        raise ValueError(f"No planet named '{name}' in the API.")
    return data

def _summarize_planet(data: dict) -> str:
    """Build a concise fact sheet from the JSON payload."""
    name = data["englishName"]
    radius_km = data.get("meanRadius")
    gravity = data.get("gravity")           # m/s²
    sidereal_day = data.get("sideralRotation")   # hours
    orbital_period = data.get("sideralOrbit")    # days
    moons = len(data.get("moons") or [])

    facts = [
        f"**{name}** quick facts:",
        f"• Mean radius: **{radius_km:,} km**"            if radius_km else "",
        f"• Surface gravity: **{gravity} m/s²**"          if gravity else "",
        f"• Sidereal day: **{sidereal_day} h**"           if sidereal_day else "",
        f"• Orbital period about the Sun: "
        f"**{orbital_period} days**"                     if orbital_period else "",
        f"• Number of known moons: **{moons}**",
        f"Data source: Le Système Solaire API"
    ]
    # Keep only the fields that exist and join non-empty strings with newlines
    return "\n".join(filter(bool, facts))
