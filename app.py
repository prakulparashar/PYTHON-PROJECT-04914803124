import streamlit as st
import folium
from streamlit_folium import st_folium # Updated to remove deprecation warning
from folium.plugins import HeatMap
import osmnx as ox
import networkx as nx
import pandas as pd
from src.audit_engine import get_city_network, get_pois
from groq import Groq




# 1. Page Configuration
st.set_page_config(page_title="Delhi 15-Min Audit", layout="wide")
st.title("🏙️ Delhi 15-Minute City Dashboard")

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
    """Sends a summarized audit report to Groq for professional review."""
    prompt = f"""
    You are an expert Urban Planner. Analyze this 15-minute city audit for {dist}:
    - Service: {amen}
    - Average Walk Time: {avg_t:.1f} minutes
    - 15-Minute Access Score: {access_p:.1f}%
    
    Provide a concise, 3-sentence professional evaluation. Suggest one specific 
    improvement for Delhi's infrastructure based on these results.
    """
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a senior consultant for the Delhi Development Authority."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return completion.choices[0].message.content

# Initialize session state for comparison
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = []

# 2. Sidebar
st.sidebar.header("Audit Settings")
# Expanded Delhi Districts
delhi_districts = [
    "Central Delhi", "Central North Delhi", "East Delhi", "New Delhi", 
    "North Delhi", "North East Delhi", "North West Delhi", "Old Delhi", 
    "Outer North Delhi", "South Delhi", "South East Delhi", "South West Delhi", 
    "West Delhi"
]

district = st.sidebar.selectbox("Select Delhi District", delhi_districts)
amenity = st.sidebar.selectbox("Essential Service", 
                                ["hospital", "school", "supermarket", "pharmacy"])

if st.sidebar.button("Clear Chat Memory"):
    st.session_state.messages = [{"role": "system", "content": "You are a Delhi urban planning expert."}]
    st.sidebar.success("Chat history cleared!")

# 3. Tabs
tab1, tab2, tab3 = st.tabs(["📍 Live Audit", "📊 Benchmarking", "💬 Urban Planning Chat"])

with tab1:
    # 1. THE ACTION: When the button is clicked, we calculate and save to state
    if st.sidebar.button("Run Audit"):
        with st.spinner(f"Analyzing {district}..."):
            # A. Fetch Data
            full_place_name = f"{district}, Delhi, India"
            G = get_city_network(full_place_name)
            pois = get_pois(full_place_name, amenity)
            
            if pois.empty:
                st.error(f"❌ No '{amenity}' found.")
                st.stop() 
            
            # B. Math & Calculations
            target_nodes = ox.distance.nearest_nodes(G, pois.geometry.x, pois.geometry.y)
            distances = nx.multi_source_dijkstra_path_length(G, set(target_nodes), weight='time')
            avg_time = sum(distances.values()) / len(distances)
            percent_served = (sum(1 for t in distances.values() if t <= 15) / len(distances)) * 100

            # C. Prepare Dataframe 
            display_df = pois[['name', 'geometry']].copy()
            display_df['Latitude'] = display_df.geometry.y
            display_df['Longitude'] = display_df.geometry.x
            display_df = display_df.drop(columns=['geometry']).fillna("Unnamed Location")

            # D. Update Chatbot Context
            audit_summary = (f"{district} {amenity} analysis: Avg walk {avg_time:.1f}m, 15-min access {percent_served:.1f}%.")
            st.session_state.messages[0] = {"role": "system", "content": f"You are a Delhi expert. Context: {audit_summary}"}

            # E. SAVE EVERYTHING TO SESSION STATE
            st.session_state.audit_results = {
                "avg_time": avg_time,
                "percent_served": percent_served,
                "display_df": display_df,
                "distances": distances,
                "pois": pois,
                "G": G,
                "district": district,
                "amenity": amenity
            }
            
            # Save for Tab 2 Comparison
            st.session_state.comparison_data.append({
                "District": district, "Service": amenity.title(),
                "Avg Walk (Min)": round(avg_time, 2), "15-Min Access %": round(percent_served, 2)
            })

    # 2. THE DISPLAY: This part runs every time the page refreshes (e.g., when switching tabs)
    if st.session_state.audit_results:
        res = st.session_state.audit_results
        
        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("Avg. Walk Time", f"{res['avg_time']:.1f} mins")
        col2.metric("15-Min Access %", f"{res['percent_served']:.1f}%")
        
        # Map
        m = folium.Map(location=[res['pois'].geometry.y.iloc[0], res['pois'].geometry.x.iloc[0]], zoom_start=14)
        HeatMap([[res['G'].nodes[n]['y'], res['G'].nodes[n]['x'], max(0, 15-t)] for n, t in res['distances'].items() if t <= 20]).add_to(m)
        st_folium(m, width=1100, height=500, returned_objects=[])

        # Histogram
        st.subheader("🚶 Walking Time Distribution")
        st.bar_chart(pd.DataFrame(list(res['distances'].values()), columns=['Minutes']), color="#2ecc71")
        
        # Log Table
        st.subheader(f"📊 Audit Log: {res['amenity'].title()}")
        st.dataframe(res['display_df'], width="stretch")
        
        # AI Insight (Calls the function using the saved results)
        st.divider()
        st.subheader("🤖 AI Urban Planner Insights")
        with st.status("Reviewing results..."):
            st.write(get_ai_insight(res['district'], res['amenity'], res['avg_time'], res['percent_served']))
    else:
        st.info("👈 Set your district and click 'Run Audit' in the sidebar to begin.")
        

            

with tab2:
    st.header("Comparative Analysis")
    if st.session_state.comparison_data:
        df_comp = pd.DataFrame(st.session_state.comparison_data).drop_duplicates()
        st.subheader("Performance Scorecard")
        styled_df = df_comp.style.highlight_max(axis=0, subset=['15-Min Access %'], color='#90ee90') \
                                .background_gradient(cmap='RdYlGn', subset=['15-Min Access %'], vmin=0, vmax=100) \
                                .format(precision=2)
        st.table(styled_df)
        
        st.subheader("Visual Benchmarking")
        st.bar_chart(df_comp.set_index('District')['15-Min Access %'])
        
        if st.button("Reset Comparison Chart"):
            st.session_state.comparison_data = []
            st.rerun()
    else:
        st.info("Run an audit in Tab 1 to see data here.")



with tab3:
    st.header("💬 Urban Planning Assistant")
    st.info("Ask me about service gaps, infrastructure suggestions, or specific district comparisons.")

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can we improve walking access in Delhi?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # We pass the entire message history to maintain context
            response_container = st.empty()
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=st.session_state.messages,
                stream=True
            )
            full_response = st.write_stream(stream)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})