import osmnx as ox
import os
import streamlit as st
import pandas as pd
import geopandas as gpd 
from shapely import wkt 


os.makedirs("data/networks", exist_ok=True)
os.makedirs("data/pois", exist_ok=True)

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



