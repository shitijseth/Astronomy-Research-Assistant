# ============================
# astronomy_mcp_server.py
# Enhanced MCP server for Astronomy Research Assistant
# Requires: Python >=3.10, FastMCP, requests, astroquery, pandas, numpy, astropy
# Optional: python-dotenv, matplotlib, pillow
# ============================

from __future__ import annotations
import os
import json
import textwrap
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from io import BytesIO

# Prevent TensorFlow/CUDA conflicts in Colab, but allow PyTorch GPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
# Note: We DON'T set CUDA_VISIBLE_DEVICES = '-1' to allow PyTorch GPU usage
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import requests
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

try:
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
    from astroquery.skyview import SkyView
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.time import Time
    ASTROPY_AVAILABLE = True
    print("[INFO] Astroquery/Astropy loaded successfully")
except ImportError:
    ASTROPY_AVAILABLE = False
    print("[WARNING] Astroquery/Astropy not available. Some features will be limited.")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
    print("[INFO] Matplotlib loaded successfully")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARNING] Matplotlib not available. Plotting features disabled.")

# Initialize MCP server
mcp = FastMCP("EnhancedAstronomyAssistant")
print("[SERVER] Enhanced Astronomy MCP server loaded")

# API Keys from environment
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# ============================================================================
# NASA & SPACE AGENCY DATA TOOLS
# ============================================================================

@mcp.tool()
def get_apod(date: Optional[str] = None, hd: bool = False) -> str:
    """Get NASA Astronomy Picture of the Day with enhanced details.
    
    Parameters:
    - date: YYYY-MM-DD format (optional, defaults to today)
    - hd: If True, includes HD image URL
    """
    params = {"api_key": NASA_API_KEY}
    if date:
        params["date"] = date
    if hd:
        params["hd"] = "True"

    try:
        resp = requests.get("https://api.nasa.gov/planetary/apod", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        result = {
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "explanation": data.get("explanation", ""),
            "media_type": data.get("media_type", ""),
            "url": data.get("url", ""),
            "copyright": data.get("copyright", "Public Domain")
        }
        
        if hd and "hdurl" in data:
            result["hd_url"] = data["hdurl"]
            
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching APOD: {str(e)}"

@mcp.tool()
def get_neo_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Get Near Earth Objects (asteroids) data from NASA.
    
    Parameters:
    - start_date: YYYY-MM-DD format
    - end_date: YYYY-MM-DD format (max 7 days from start_date)
    """
    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=7)).strftime("%Y-%m-%d")
    
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": NASA_API_KEY
    }
    
    try:
        resp = requests.get("https://api.nasa.gov/neo/rest/v1/feed", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        
        neo_count = data.get("element_count", 0)
        result = f"Near Earth Objects from {start_date} to {end_date}: {neo_count} objects\n\n"
        
        for date, neos in data.get("near_earth_objects", {}).items():
            result += f"Date: {date}\n"
            for neo in neos[:3]:  # Show top 3 per date
                name = neo.get("name", "Unknown")
                diameter = neo.get("estimated_diameter", {}).get("meters", {})
                diameter_str = f"{diameter.get('estimated_diameter_min', 0):.1f}-{diameter.get('estimated_diameter_max', 0):.1f}m"
                hazardous = "⚠️ HAZARDOUS" if neo.get("is_potentially_hazardous_asteroid") else "Safe"
                result += f"  • {name} ({diameter_str}) - {hazardous}\n"
            result += "\n"
            
        return result
    except Exception as e:
        return f"Error fetching NEO data: {str(e)}"

@mcp.tool()
def get_mars_weather() -> str:
    """Get latest Mars weather data from NASA InSight mission."""
    try:
        resp = requests.get(f"https://api.nasa.gov/insight_weather/?api_key={NASA_API_KEY}&feedtype=json&ver=1.0", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        # Get most recent sol (Martian day)
        sols = list(data.keys())
        sols = [s for s in sols if s.isdigit()]
        if not sols:
            return "No recent Mars weather data available"
        
        latest_sol = max(sols, key=int)
        weather = data[latest_sol]
        
        result = f"Mars Weather (Sol {latest_sol}):\n"
        if "AT" in weather:
            temp = weather["AT"]
            result += f"Temperature: {temp.get('av', 'N/A')}°C (avg), {temp.get('mn', 'N/A')}°C (min), {temp.get('mx', 'N/A')}°C (max)\n"
        if "PRE" in weather:
            pressure = weather["PRE"]
            result += f"Pressure: {pressure.get('av', 'N/A')} Pa (avg)\n"
        if "WD" in weather:
            wind = weather["WD"]
            result += f"Wind: Most common direction {wind.get('most_common', {}).get('compass_point', 'N/A')}\n"
        
        return result
    except Exception as e:
        return f"Error fetching Mars weather: {str(e)}"

# ============================================================================
# EXOPLANET & STELLAR DATA TOOLS
# ============================================================================

@mcp.tool()
def search_exoplanets(criteria: str = "habitable", limit: int = 10) -> str:
    """Search for exoplanets based on various criteria.
    
    Parameters:
    - criteria: "habitable", "recent", "small", "large", "close" or custom query
    - limit: Maximum number of results
    """
    if not ASTROPY_AVAILABLE:
        return "Astroquery not available. Cannot search exoplanet database."
    
    try:
        if criteria == "habitable":
            # Potentially habitable planets (in habitable zone with Earth-like size)
            table = NasaExoplanetArchive.query_criteria(
                table="ps",
                where="pl_rade<2 and pl_orbsmax>0.5 and pl_orbsmax<2 and st_teff>4000 and st_teff<7000",
                order="pl_rade",
                select="pl_name,hostname,pl_rade,pl_orbper,pl_eqt,st_dist,disc_year"
            )
        elif criteria == "recent":
            table = NasaExoplanetArchive.query_criteria(
                table="ps",
                where="disc_year>2020",
                order="disc_year desc",
                select="pl_name,hostname,pl_rade,pl_masse,disc_year,discoverymethod"
            )
        elif criteria == "small":
            table = NasaExoplanetArchive.query_criteria(
                table="ps",
                where="pl_rade<1.5",
                order="pl_rade",
                select="pl_name,hostname,pl_rade,pl_orbper,st_dist"
            )
        elif criteria == "large":
            table = NasaExoplanetArchive.query_criteria(
                table="ps",
                where="pl_rade>10",
                order="pl_rade desc",
                select="pl_name,hostname,pl_rade,pl_masse,pl_orbper"
            )
        elif criteria == "close":
            table = NasaExoplanetArchive.query_criteria(
                table="ps",
                where="st_dist<50",
                order="st_dist",
                select="pl_name,hostname,st_dist,pl_rade,disc_year"
            )
        else:
            return f"Unknown criteria: {criteria}. Use: habitable, recent, small, large, or close"
        
        if len(table) == 0:
            return f"No exoplanets found matching criteria: {criteria}"
        
        # Limit results
        table = table[:limit]
        
        result = f"Found {len(table)} exoplanets matching '{criteria}':\n\n"
        for row in table:
            name = row.get('pl_name', 'N/A')
            host = row.get('hostname', 'N/A')
            radius = row.get('pl_rade', 'N/A')
            period = row.get('pl_orbper', 'N/A')
            distance = row.get('st_dist', 'N/A')
            year = row.get('disc_year', 'N/A')
            
            result += f"• {name} (around {host})\n"
            result += f"  Radius: {radius} Earth radii, Period: {period} days\n"
            if distance != 'N/A':
                result += f"  Distance: {distance} parsecs\n"
            if year != 'N/A':
                result += f"  Discovered: {year}\n"
            result += "\n"
        
        return result
    except Exception as e:
        return f"Error searching exoplanets: {str(e)}"

@mcp.tool()
def get_star_info(star_name: str) -> str:
    """Get detailed information about a star from SIMBAD database."""
    if not ASTROPY_AVAILABLE:
        return "Astroquery not available. Cannot query stellar database."
    
    try:
        # Configure SIMBAD to get more fields
        Simbad.add_votable_fields('sptype', 'parallax', 'pmra', 'pmdec', 'rv_value', 'flux(V)')
        
        result_table = Simbad.query_object(star_name)
        if result_table is None or len(result_table) == 0:
            return f"Star '{star_name}' not found in SIMBAD database"
        
        star = result_table[0]
        
        result = f"Star Information: {star_name}\n"
        result += f"Main Identifier: {star['MAIN_ID']}\n"
        result += f"Object Type: {star['OTYPE']}\n"
        result += f"Coordinates: RA {star['RA']}, Dec {star['DEC']}\n"
        
        if not star['SP_TYPE'].mask:
            result += f"Spectral Type: {star['SP_TYPE']}\n"
        if not star['PLX_VALUE'].mask:
            distance = 1000 / star['PLX_VALUE']  # Convert parallax to distance in parsecs
            result += f"Distance: ~{distance:.1f} parsecs ({distance * 3.26:.1f} light-years)\n"
        if not star['FLUX_V'].mask:
            result += f"Visual Magnitude: {star['FLUX_V']:.2f}\n"
        if not star['RV_VALUE'].mask:
            result += f"Radial Velocity: {star['RV_VALUE']:.1f} km/s\n"
        
        return result
    except Exception as e:
        return f"Error querying star database: {str(e)}"

# ============================================================================
# OBSERVING & IMAGING TOOLS
# ============================================================================

@mcp.tool()
def get_sky_image(target: str, survey: str = "DSS", size: str = "0.5 deg") -> str:
    """Get astronomical sky image of a target object.
    
    Parameters:
    - target: Object name (e.g., "M31", "NGC 1300", "Orion Nebula")
    - survey: "DSS", "2MASS-J", "WISE 3.4" or other SkyView survey
    - size: Angular size (e.g., "0.5 deg", "10 arcmin")
    """
    if not ASTROPY_AVAILABLE:
        return "Astroquery not available. Cannot retrieve sky images."
    
    try:
        # Get image from SkyView
        img_list = SkyView.get_images(position=target, survey=survey, pixels=800, 
                                     radius=size, cache=False)
        
        if not img_list:
            return f"No image found for {target} in {survey} survey"
        
        # Convert to base64 for display (simplified - in practice you'd save to file)
        result = f"Sky image retrieved for {target}\n"
        result += f"Survey: {survey}\n"
        result += f"Field size: {size}\n"
        result += f"Image dimensions: {img_list[0].data.shape}\n"
        result += "Use appropriate software to display the FITS image data."
        
        return result
    except Exception as e:
        return f"Error retrieving sky image: {str(e)}"

@mcp.tool()
def get_observing_conditions(latitude: float, longitude: float) -> str:
    """Get current observing conditions for a location.
    
    Parameters:
    - latitude: Decimal degrees
    - longitude: Decimal degrees
    """
    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key not configured. Cannot get weather data."
    
    try:
        # Get weather data
        weather_url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        resp = requests.get(weather_url, params=params, timeout=10)
        resp.raise_for_status()
        weather = resp.json()
        
        result = f"Observing Conditions for {latitude:.2f}°, {longitude:.2f}°:\n\n"
        result += f"Location: {weather.get('name', 'Unknown')}\n"
        result += f"Weather: {weather['weather'][0]['description'].title()}\n"
        result += f"Temperature: {weather['main']['temp']:.1f}°C\n"
        result += f"Humidity: {weather['main']['humidity']}%\n"
        result += f"Pressure: {weather['main']['pressure']} hPa\n"
        result += f"Wind: {weather['wind'].get('speed', 0):.1f} m/s\n"
        result += f"Cloud Cover: {weather['clouds']['all']}%\n"
        result += f"Visibility: {weather.get('visibility', 'N/A')} m\n"
        
        # Basic observing assessment
        clouds = weather['clouds']['all']
        if clouds < 20:
            result += "\n🌟 Excellent observing conditions!"
        elif clouds < 50:
            result += "\n⭐ Good observing conditions"
        elif clouds < 80:
            result += "\n☁️ Fair conditions, some clouds"
        else:
            result += "\n☁️ Poor conditions, mostly cloudy"
        
        return result
    except Exception as e:
        return f"Error getting observing conditions: {str(e)}"

# ============================================================================
# EDUCATIONAL & CALCULATION TOOLS
# ============================================================================

@mcp.tool()
def calculate_planet_weight(weight_kg: float, planet: str) -> str:
    """Calculate your weight on different planets and moons.
    
    Parameters:
    - weight_kg: Your weight on Earth in kilograms
    - planet: Planet name (Earth, Mars, Moon, Jupiter, etc.)
    """
    # Surface gravity relative to Earth
    gravity_factors = {
        "mercury": 0.378,
        "venus": 0.907,
        "earth": 1.0,
        "mars": 0.377,
        "jupiter": 2.36,
        "saturn": 0.916,
        "uranus": 0.889,
        "neptune": 1.13,
        "moon": 0.166,
        "sun": 27.01,
        "pluto": 0.071
    }
    
    planet_lower = planet.lower()
    if planet_lower not in gravity_factors:
        return f"Unknown planet: {planet}. Available: {', '.join(gravity_factors.keys())}"
    
    new_weight = weight_kg * gravity_factors[planet_lower]
    
    result = f"Weight Comparison:\n"
    result += f"Your weight on Earth: {weight_kg:.1f} kg\n"
    result += f"Your weight on {planet.title()}: {new_weight:.1f} kg\n"
    result += f"Gravity factor: {gravity_factors[planet_lower]:.3f} × Earth gravity\n"
    
    if new_weight > weight_kg:
        result += f"You would feel {new_weight/weight_kg:.1f}× heavier!"
    else:
        result += f"You would feel {weight_kg/new_weight:.1f}× lighter!"
    
    return result

@mcp.tool()
def calculate_light_travel(distance_ly: float) -> str:
    """Calculate how long light takes to travel astronomical distances.
    
    Parameters:
    - distance_ly: Distance in light-years
    """
    # Convert to various time units
    days = distance_ly * 365.25
    hours = days * 24
    minutes = hours * 60
    seconds = minutes * 60
    
    result = f"Light Travel Time for {distance_ly} light-years:\n\n"
    
    if distance_ly < 1:
        if days < 1:
            if hours < 1:
                if minutes < 1:
                    result += f"Time: {seconds:.1f} seconds"
                else:
                    result += f"Time: {minutes:.1f} minutes"
            else:
                result += f"Time: {hours:.1f} hours"
        else:
            result += f"Time: {days:.1f} days"
    else:
        result += f"Time: {distance_ly:.1f} years"
    
    # Add context examples
    if distance_ly < 0.001:
        result += "\n(Less than the distance to the Moon)"
    elif distance_ly < 0.1:
        result += "\n(Within our solar system)"
    elif distance_ly < 10:
        result += "\n(Nearby stars)"
    elif distance_ly < 1000:
        result += "\n(Within our galaxy)"
    else:
        result += "\n(Intergalactic distances)"
    
    return result

@mcp.tool()
def generate_constellation_info(constellation: str) -> str:
    """Get information about a constellation including main stars and mythology.
    
    Parameters:
    - constellation: Constellation name (e.g., "Orion", "Ursa Major")
    """
    # Basic constellation data (in a real implementation, this would come from a database)
    constellations = {
        "orion": {
            "name": "Orion",
            "meaning": "The Hunter",
            "main_stars": ["Betelgeuse", "Rigel", "Bellatrix", "Mintaka", "Alnilam", "Alnitak"],
            "mythology": "In Greek mythology, Orion was a great hunter. The constellation depicts him with a belt and sword.",
            "best_visible": "Winter (Northern Hemisphere)",
            "notable_objects": ["Orion Nebula (M42)", "Horsehead Nebula", "Flame Nebula"]
        },
        "ursa major": {
            "name": "Ursa Major",
            "meaning": "Great Bear",
            "main_stars": ["Dubhe", "Merak", "Phecda", "Megrez", "Alioth", "Mizar", "Alkaid"],
            "mythology": "The Great Bear, containing the Big Dipper asterism. In Greek myth, it represents Callisto.",
            "best_visible": "Spring (Northern Hemisphere)",
            "notable_objects": ["M81 (Bode's Galaxy)", "M82 (Cigar Galaxy)", "Whirlpool Galaxy (M51)"]
        },
        "cassiopeia": {
            "name": "Cassiopeia",
            "meaning": "The Queen",
            "main_stars": ["Schedar", "Caph", "Gamma Cas", "Ruchbah", "Segin"],
            "mythology": "Named after the vain queen Cassiopeia in Greek mythology.",
            "best_visible": "Autumn (Northern Hemisphere)",
            "notable_objects": ["Heart Nebula", "Soul Nebula", "Pacman Nebula"]
        }
    }
    
    const_key = constellation.lower()
    if const_key not in constellations:
        return f"Constellation '{constellation}' not found in database. Try: Orion, Ursa Major, Cassiopeia"
    
    info = constellations[const_key]
    
    result = f"Constellation: {info['name']} ({info['meaning']})\n\n"
    result += f"Mythology: {info['mythology']}\n\n"
    result += f"Best visible: {info['best_visible']}\n\n"
    result += f"Main stars: {', '.join(info['main_stars'])}\n\n"
    result += f"Notable deep-sky objects: {', '.join(info['notable_objects'])}\n"
    
    return result

# ============================================================================
# SPACE MISSION & NEWS TOOLS
# ============================================================================

@mcp.tool()
def get_iss_location() -> str:
    """Get current location and details of the International Space Station."""
    try:
        # Get current ISS position
        resp = requests.get("http://api.open-notify.org/iss-now.json", timeout=10)
        resp.raise_for_status()
        iss_data = resp.json()
        
        lat = float(iss_data['iss_position']['latitude'])
        lon = float(iss_data['iss_position']['longitude'])
        timestamp = iss_data['timestamp']
        
        # Get people in space
        people_resp = requests.get("http://api.open-notify.org/astros.json", timeout=10)
        people_resp.raise_for_status()
        people_data = people_resp.json()
        
        result = f"International Space Station Status:\n\n"
        result += f"Current Position: {lat:.2f}°N, {lon:.2f}°E\n"
        result += f"Timestamp: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        result += f"Altitude: ~408 km (253 miles)\n"
        result += f"Speed: ~27,600 km/h (17,150 mph)\n\n"
        
        # People currently in space
        iss_crew = [person for person in people_data['people'] if person['craft'] == 'ISS']
        result += f"Current ISS Crew ({len(iss_crew)} people):\n"
        for person in iss_crew:
            result += f"• {person['name']}\n"
        
        return result
    except Exception as e:
        return f"Error getting ISS data: {str(e)}"

@mcp.tool()
def get_spacex_launches(limit: int = 5) -> str:
    """Get recent and upcoming SpaceX launches.
    
    Parameters:
    - limit: Number of launches to return
    """
    try:
        # Get latest launches
        resp = requests.get(f"https://api.spacexdata.com/v4/launches/latest?limit={limit}", timeout=15)
        resp.raise_for_status()
        launches = resp.json() if isinstance(resp.json(), list) else [resp.json()]
        
        # Get upcoming launches
        upcoming_resp = requests.get(f"https://api.spacexdata.com/v4/launches/upcoming?limit={limit}", timeout=15)
        upcoming_resp.raise_for_status()
        upcoming = upcoming_resp.json()
        
        result = "SpaceX Launch Information:\n\n"
        
        result += "Recent Launches:\n"
        for launch in launches[:limit]:
            name = launch.get('name', 'Unknown')
            date = launch.get('date_local', 'Unknown')
            success = "✅ Success" if launch.get('success') else "❌ Failed"
            details = launch.get('details', 'No details available')[:100] + "..."
            
            result += f"• {name} ({date})\n"
            result += f"  Status: {success}\n"
            result += f"  Details: {details}\n\n"
        
        result += "Upcoming Launches:\n"
        for launch in upcoming[:limit]:
            name = launch.get('name', 'Unknown')
            date = launch.get('date_local', 'TBD')
            details = launch.get('details', 'No details available')
            if details and len(details) > 100:
                details = details[:100] + "..."
            
            result += f"• {name} ({date})\n"
            result += f"  Details: {details}\n\n"
        
        return result
    except Exception as e:
        return f"Error getting SpaceX launch data: {str(e)}"

# ============================================================================
# ENHANCED SOLAR SYSTEM RESOURCES
# ============================================================================

@mcp.resource("planet://{name}")
def get_planet_resource(name: str) -> str:
    """Enhanced planet resource with more detailed information."""
    try:
        data = _fetch_planet_json(name)
        return _detailed_planet_summary(data)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error fetching planet data: {e}"

def _fetch_planet_json(name: str) -> dict:
    """Fetch planet data from Solar System API."""
    url = f"https://api.le-systeme-solaire.net/rest/bodies/{name.lower()}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise ValueError(f"Could not retrieve data for '{name}'. Status code: {r.status_code}")
    data = r.json()
    if data.get("englishName", "").lower() != name.lower():
        raise ValueError(f"No planet named '{name}' in the API.")
    return data

def _detailed_planet_summary(data: dict) -> str:
    """Generate detailed planet summary."""
    name = data["englishName"]
    
    # Basic properties
    result = f"# {name} - Detailed Profile\n\n"
    
    # Physical characteristics
    result += "## Physical Characteristics\n"
    if data.get("meanRadius"):
        result += f"• **Mean Radius**: {data['meanRadius']:,} km\n"
        earth_radii = data['meanRadius'] / 6371
        result += f"  ({earth_radii:.2f} Earth radii)\n"
    
    if data.get("mass"):
        result += f"• **Mass**: {data['mass']['massValue']:.2e} × 10^{data['mass']['massExponent']} kg\n"
    
    if data.get("density"):
        result += f"• **Density**: {data['density']:.2f} g/cm³\n"
    
    if data.get("gravity"):
        result += f"• **Surface Gravity**: {data['gravity']:.2f} m/s²\n"
        earth_gravity = data['gravity'] / 9.81
        result += f"  ({earth_gravity:.2f} × Earth gravity)\n"
    
    # Orbital characteristics
    result += "\n## Orbital Characteristics\n"
    if data.get("semimajorAxis"):
        au = data['semimajorAxis'] / 149597870.7  # Convert km to AU
        result += f"• **Distance from Sun**: {data['semimajorAxis']:,} km ({au:.2f} AU)\n"
    
    if data.get("sideralOrbit"):
        result += f"• **Orbital Period**: {data['sideralOrbit']:.1f} Earth days\n"
        if data['sideralOrbit'] > 365:
            years = data['sideralOrbit'] / 365.25
            result += f"  ({years:.2f} Earth years)\n"
    
    if data.get("eccentricity"):
        result += f"• **Orbital Eccentricity**: {data['eccentricity']:.4f}\n"
    
    # Rotation
    if data.get("sideralRotation"):
        result += f"• **Rotation Period**: {data['sideralRotation']:.2f} hours\n"
        if abs(data['sideralRotation']) > 24:
            days = data['sideralRotation'] / 24
            result += f"  ({days:.2f} Earth days)\n"
    
    # Atmosphere
    if data.get("atmosphere"):
        result += "\n## Atmospheric Composition\n"
        for component, percentage in data['atmosphere'].items():
            if component != "composition":
                result += f"• **{component}**: {percentage}%\n"
    
    # Moons
    moons = data.get("moons", [])
    if moons:
        result += f"\n## Moons ({len(moons)} total)\n"
        if len(moons) <= 10:
            for moon in moons:
                result += f"• {moon}\n"
        else:
            for moon in moons[:5]:
                result += f"• {moon}\n"
            result += f"• ... and {len(moons) - 5} more\n"
    
    result += f"\n*Data source: Le Système Solaire API*"
    
    return result

if __name__ == "__main__":
    print("[SERVER] Starting MCP server...")
    print("[SERVER] Server will stay running until stopped with Ctrl+C")
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("[SERVER] Server stopped by user")
    except Exception as e:
        print(f"[SERVER] Server error: {e}")