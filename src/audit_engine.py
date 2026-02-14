import osmnx as ox
import os
import streamlit as st
import pandas as pd
import geopandas as gpd 
from shapely import wkt 


os.makedirs("data/networks", exist_ok=True)
os.makedirs("data/pois", exist_ok=True)



# ─────────────────────────────────────────────────────────────
# POPULATION-BASED FACILITY STANDARDS
# Sources: WHO, MoHFW India, UDPFI Guidelines, Delhi Master Plan
# ─────────────────────────────────────────────────────────────
FACILITY_STANDARDS = {
    "hospital": {
        "label": "Hospitals / Clinics",
        "per_population": 10_000,       # 1 per 10,000 residents (MoHFW)
        "unit": "facility",
    },
    "school": {
        "label": "Schools",
        "per_population": 5_000,        # 1 per 5,000 residents (UDPFI)
        "unit": "facility",
    },
    "supermarket": {
        "label": "Grocery / Supermarkets",
        "per_population": 3_000,        # 1 per 3,000 residents
        "unit": "facility",
    },
    "pharmacy": {
        "label": "Pharmacies",
        "per_population": 4_000,        # 1 per 4,000 residents (Delhi MP)
        "unit": "facility",
    },
}

# Approximate 2024 populations for Delhi districts (Census + Delhi Economic Survey)
DELHI_DISTRICT_POPULATIONS = {
    "Central Delhi":       700_000,
    "East Delhi":        1_800_000,
    "New Delhi":           250_000,
    "North Delhi":       1_000_000,
    "North East Delhi":  2_300_000,
    "North West Delhi":  3_600_000,
    "South Delhi":       2_700_000,
    "South East Delhi":  1_800_000,
    "South West Delhi":  2_500_000,
    "West Delhi":        2_500_000,
    "Old Delhi":           600_000,
    "Central North Delhi": 900_000,
    "Outer North Delhi": 1_200_000,
}


def calculate_facility_requirements(district: str, amenity: str, actual_count: int) -> dict:
    """
    Given a district name, amenity type, and OSM-counted facilities,
    returns a dict with required, actual, gap, and status.
    """
    population = DELHI_DISTRICT_POPULATIONS.get(district, 1_000_000)
    standard = FACILITY_STANDARDS.get(amenity)

    if not standard:
        return None

    required = max(1, round(population / standard["per_population"]))
    gap = actual_count - required          # positive = surplus, negative = deficit

    if gap >= 0:
        status = "✅ Sufficient"
        status_color = "green"
    elif gap >= -required * 0.3:           # within 30% shortfall
        status = "⚠️ Marginal"
        status_color = "orange"
    else:
        status = "🔴 Deficit"
        status_color = "red"

    coverage_pct = round((actual_count / required) * 100, 1) if required > 0 else 0

    return {
        "district": district,
        "amenity": standard["label"],
        "population": population,
        "required": required,
        "actual": actual_count,
        "gap": gap,
        "coverage_pct": coverage_pct,
        "status": status,
        "status_color": status_color,
        "per_population": standard["per_population"],
    }




@st.cache_data(show_spinner=False)
def get_city_network(place_name):
    # Standardize filename
    safe_name = place_name.replace(", ", "_").replace(" ", "_")
    filepath = f"data/networks/{safe_name}.graphml"
    
    if os.path.exists(filepath):
        G = ox.load_graphml(filepath)
        # Type casting to float
        for u, v, k, data in G.edges(data=True, keys=True):
            if 'time' in data:
                data['time'] = float(data['time'])
        return G

    # --- DOWNLOAD LOGIC ---
    print(f"🔍 Attempting to fetch: {place_name}")
    try:
        # STRATEGY 1: Official Boundary
        G = ox.graph_from_place(place_name, network_type='walk')
    except Exception:
        # STRATEGY 2: Radial Buffer 
        print(f"⚠️ Polygon not found. Switching to 3km radial buffer for {place_name}...")
        G = ox.graph_from_address(place_name, dist=3000, network_type='walk')

    # Standardize and project
    G = ox.project_graph(G)
    
    # Calculate time (speed 75 m/min)
    speed_mpm = 75 
    for u, v, k, data in G.edges(data=True, keys=True):
        data['time'] = float(data['length']) / speed_mpm
    
    # Save for permanent cache
    ox.save_graphml(G, filepath)
    return G




@st.cache_data(show_spinner=False)
def get_pois(place_name, category):
    # 1. Check local cache first
    file_path = f"data/pois/{place_name.replace(',', '').replace(' ', '_')}_{category}.csv"
    
    if os.path.exists(file_path):
        from shapely import wkt
        import geopandas as gpd
        df = pd.read_csv(file_path)
        df['geometry'] = df['geometry'].apply(wkt.loads)
        return gpd.GeoDataFrame(df, crs="EPSG:4326")

    # 2. Define tags
    if category == 'supermarket':
        tags = {'shop': ['supermarket', 'convenience', 'grocery']}
    elif category == 'hospital':
        tags = {'amenity': ['hospital', 'clinic', 'doctors']}
    else:
        tags = {'amenity': category}

    # 3. Fetch Data with Fallback
    try:
        # Strategy A: Search within official boundary
        pois = ox.features_from_place(place_name, tags)
    except Exception:
        # Strategy B: Search within 3km of the center (for new districts)
        st.info(f"📍 Fetching {category} within 3km of {place_name} center...")
        pois = ox.features_from_address(place_name, tags, dist=3000)
        
    # 4. Clean and Save
    pois_points = pois[pois.geometry.type == 'Point'].copy()
    pois_polygons = pois[pois.geometry.type == 'Polygon'].copy()
    
    if not pois_polygons.empty:
        pois_polygons = pois_polygons.copy()
        pois_polygons['geometry'] = pois_polygons.geometry.centroid
        
        if pois_points.empty:
            # No points at all — use polygon centroids directly
            pois_points = pois_polygons
        else:
            # Combine both, reset index to avoid duplicate index issues
            pois_points = gpd.GeoDataFrame(
                pd.concat([pois_points, pois_polygons], ignore_index=True),
                crs=pois.crs
            )

    if not pois_points.empty:
        pois_save = pois_points[['name', 'geometry']].fillna("Unnamed")
        pois_save.to_csv(file_path, index=False)
        return pois_points

    return pois



