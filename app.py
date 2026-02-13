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

# 3. Tabs
tab1, tab2, tab3 = st.tabs(["📍 Live Audit", "📊 Benchmarking", "💬 Urban Planning Chat"])

with tab1:
    if st.sidebar.button("Run Audit"):
        with st.spinner(f"Analyzing {district}..."):
            # A. Fetch Data
            full_place_name = f"{district}, Delhi, India"
            G = get_city_network(full_place_name)
            pois = get_pois(full_place_name, amenity)
            
            if pois.empty:
                st.error(f"❌ No '{amenity}' found in {district}.")
                st.stop() 
            
            # B. Math Logic
            target_nodes = ox.distance.nearest_nodes(G, pois.geometry.x, pois.geometry.y)
            distances = nx.multi_source_dijkstra_path_length(G, set(target_nodes), weight='time')
            
            # C. Calculations
            avg_time = sum(distances.values()) / len(distances)
            percent_served = (sum(1 for t in distances.values() if t <= 15) / len(distances)) * 100
            
            # Save data for Tab 2
            st.session_state.comparison_data.append({
                "District": district,
                "Service": amenity.title(),
                "Avg Walk (Min)": round(avg_time, 2),
                "15-Min Access %": round(percent_served, 2)
            })

            # D. Display Metrics
            col1, col2 = st.columns(2)
            col1.metric("Avg. Walk Time", f"{avg_time:.1f} mins")
            col2.metric("15-Min Access %", f"{percent_served:.1f}%")
            
            # E. Map (Updated to remove warning)
            m = folium.Map(location=[pois.geometry.y.iloc[0], pois.geometry.x.iloc[0]], 
                           zoom_start=14, tiles="OpenStreetMap")
            
            heat_data = [[G.nodes[node]['y'], G.nodes[node]['x'], max(0, 15 - time)] 
                         for node, time in distances.items() if time <= 20]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            for _, row in pois.iterrows():
                folium.CircleMarker([row.geometry.y, row.geometry.x], radius=3, color='blue').add_to(m)
            
            # Use st_folium with returned_objects=[] to emulate static behavior
            st_folium(m, width=1100, height=500, returned_objects=[])



            st.subheader("🚶 Walking Time Distribution")

            # F. Histogram
            # 1. Prepare data for the histogram
            dist_values = list(distances.values())
            dist_df = pd.DataFrame(dist_values, columns=['Minutes'])

            # 2. Create bins (0-5, 5-10, 10-15, 15-20, 20+)
            bins = [0, 5, 10, 15, 20, 100]
            labels = ['0-5m', '5-10m', '10-15m', '15-20m', '20m+']
            dist_df['Range'] = pd.cut(dist_df['Minutes'], bins=bins, labels=labels)

            # 3. Count occurrences and plot
            chart_data = dist_df['Range'].value_counts().reindex(labels)
            st.bar_chart(chart_data, color="#2ecc71")

            st.caption("This chart shows how many areas in the district fall within each walking time bracket.")

            # G. Audit Log Table (Updated 'width' to remove warning)
            st.divider()
            st.subheader(f"📊 Audit Log: {amenity.title()} Locations in {district}")

            display_df = pois[['name', 'geometry']].copy()
            display_df['Latitude'] = display_df.geometry.y
            display_df['Longitude'] = display_df.geometry.x
            display_df = display_df.drop(columns=['geometry']).fillna("Unnamed Location")

            st.dataframe(display_df, width="stretch")

            # H. Download Data
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {district} {amenity.title()} Data",
                data=csv,
                file_name=f'delhi_audit_{district}_{amenity}.csv',
                mime='text/csv',
            )

            # I. AI insight
            st.divider()
            st.subheader("🤖 AI Urban Planner Insights")
            with st.status("Generating spatial analysis...", expanded=True):
                ai_review = get_ai_insight(district, amenity, avg_time, percent_served)
                st.write(ai_review)

            

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