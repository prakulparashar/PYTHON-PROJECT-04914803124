import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import osmnx as ox
import networkx as nx
import pandas as pd
from src.audit_engine import get_city_network, get_pois
from groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="Delhi 15-Min Audit", layout="wide")
st.title("🏙️ Delhi 15-Minute City Dashboard")

# --- CACHING WRAPPERS (Add these to stop the lag) ---
@st.cache_data(show_spinner=False)
def cached_network(place):
    return get_city_network(place)

@st.cache_data(show_spinner=False)
def cached_pois(place, amenity):
    return get_pois(place, amenity)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a Delhi urban planning expert."}
    ]

if 'audit_results' not in st.session_state:
    st.session_state.audit_results = None

if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = []

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_ai_insight(dist, amen, avg_t, access_p):
    prompt = f"""
    You are an expert Urban Planner. Analyze this 15-minute city audit for {dist}:
    - Service: {amen}
    - Average Walk Time: {avg_t:.1f} minutes
    - 15-Minute Access Score: {access_p:.1f}%
    Provide a concise, 3-sentence evaluation and one specific infrastructure suggestion.
    """
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a senior consultant for the DDA."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return completion.choices[0].message.content

# 2. Sidebar
st.sidebar.header("Audit Settings")
delhi_districts = [
    "Central Delhi", "East Delhi", "New Delhi", "North Delhi", "South Delhi", "West Delhi"
]

district = st.sidebar.selectbox("Select Delhi District", delhi_districts)
amenity = st.sidebar.selectbox("Essential Service", ["hospital", "school", "supermarket", "pharmacy"])

if st.sidebar.button("Clear Chat Memory"):
    st.session_state.messages = [{"role": "system", "content": "You are a Delhi urban planning expert."}]
    st.sidebar.success("Chat history cleared!")

# 3. Tabs
tab1, tab2, tab3 = st.tabs(["📍 Live Audit", "📊 Benchmarking", "💬 Urban Planning Chat"])

with tab1:
    if st.sidebar.button("Run Audit"):
        with st.spinner(f"Analyzing {district}..."):
            full_place_name = f"{district}, Delhi, India"
            
            # Use Cached Functions to prevent re-downloading data on every chat message
            G = cached_network(full_place_name)
            pois = cached_pois(full_place_name, amenity)
            
            if pois.empty:
                st.error(f"❌ No '{amenity}' found.")
                st.stop() 
            
            target_nodes = ox.distance.nearest_nodes(G, pois.geometry.x, pois.geometry.y)
            distances = nx.multi_source_dijkstra_path_length(G, set(target_nodes), weight='time')
            avg_time = sum(distances.values()) / len(distances)
            percent_served = (sum(1 for t in distances.values() if t <= 15) / len(distances)) * 100

            display_df = pois[['name', 'geometry']].copy()
            display_df['Latitude'] = display_df.geometry.y
            display_df['Longitude'] = display_df.geometry.x
            display_df = display_df.drop(columns=['geometry']).fillna("Unnamed Location")

            audit_summary = (f"{district} {amenity}: Avg walk {avg_time:.1f}m, 15-min access {percent_served:.1f}%.")
            st.session_state.messages[0] = {"role": "system", "content": f"You are a Delhi expert. Context: {audit_summary}"}

            # Save to state
            st.session_state.audit_results = {
                "avg_time": avg_time, "percent_served": percent_served,
                "display_df": display_df, "distances": distances,
                "pois": pois, "G": G, "district": district, "amenity": amenity
            }
            
            st.session_state.comparison_data.append({
                "District": district, "Service": amenity.title(),
                "Avg Walk (Min)": round(avg_time, 2), "15-Min Access %": round(percent_served, 2)
            })

    # Display Persistent Results
    if st.session_state.audit_results:
        res = st.session_state.audit_results
        col1, col2 = st.columns(2)
        col1.metric("Avg. Walk Time", f"{res['avg_time']:.1f} mins")
        col2.metric("15-Min Access %", f"{res['percent_served']:.1f}%")
        
        # Map Optimization: returned_objects=[] stops the laggy "ping-pong" data transfer
        m = folium.Map(location=[res['pois'].geometry.y.iloc[0], res['pois'].geometry.x.iloc[0]], zoom_start=14)
        HeatMap([[res['G'].nodes[n]['y'], res['G'].nodes[n]['x'], max(0, 15-t)] for n, t in res['distances'].items() if t <= 20]).add_to(m)
        st_folium(m, width=1100, height=500, returned_objects=[], key="audit_map")

        st.bar_chart(pd.DataFrame(list(res['distances'].values()), columns=['Minutes']), color="#2ecc71")
        st.dataframe(res['display_df'], use_container_width=True)
    else:
        st.info("👈 Run an Audit to see spatial data.")

with tab2:
    if st.session_state.comparison_data:
        df_comp = pd.DataFrame(st.session_state.comparison_data).drop_duplicates()
        st.table(df_comp)
        st.bar_chart(df_comp.set_index('District')['15-Min Access %'])

with tab3:
    st.header("💬 Urban Planning Assistant")
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the audit..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                stream=True
            )
            full_response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})