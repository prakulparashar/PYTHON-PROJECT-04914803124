import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import random
from src.audit_engine import get_city_network, get_pois, calculate_facility_requirements
from groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="Delhi 15-Min Audit", layout="wide")
st.title("Delhi 15-Minute City Dashboard")

# --- CACHING WRAPPERS ---
@st.cache_resource(show_spinner=False)
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

if "audit_results" not in st.session_state:
    st.session_state.audit_results = None

if "comparison_data" not in st.session_state:
    st.session_state.comparison_data = []

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def get_ai_insight(dist, amen, avg_t, access_p):
    prompt = (
        f"You are an expert Urban Planner. Analyze this 15-minute city audit for {dist}:\n"
        f"- Service: {amen}\n"
        f"- Average Walk Time: {avg_t:.1f} minutes\n"
        f"- 15-Minute Access Score: {access_p:.1f}%\n"
        "Provide a concise, 3-sentence evaluation and one specific infrastructure suggestion."
    )
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a senior consultant for the DDA."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return completion.choices[0].message.content


def render_requirements_panel(req: dict):
    """Renders the population-based requirements card below the metrics."""
    population = req["population"]
    required = req["required"]
    actual = req["actual"]
    gap = req["gap"]
    coverage_pct = req["coverage_pct"]
    status = req["status"]
    per_pop = req["per_population"]

    st.markdown("---")
    st.subheader("📊 Population-Based Requirement Analysis")

    # Top info row
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric(
        "District Population",
        f"{population:,.0f}",
        help="Source: Delhi Economic Survey 2024 estimates"
    )
    col_b.metric(
        "Required Facilities",
        required,
        help=f"Standard: 1 per {per_pop:,} residents (MoHFW / UDPFI Guidelines)"
    )
    col_c.metric(
        "Mapped by OSM",
        actual,
        delta=f"{'+' if gap >= 0 else ''}{gap} vs required",
        delta_color="normal" if gap >= 0 else "inverse",
    )
    col_d.metric(
        "Coverage",
        f"{coverage_pct}%",
        help="(Actual ÷ Required) × 100"
    )

    # Progress bar
    bar_pct = min(coverage_pct / 100, 1.0)
    bar_color = "#2ecc71" if gap >= 0 else ("#f39c12" if gap >= -required * 0.3 else "#e74c3c")

    st.markdown(
        f"""
        <div style="margin: 8px 0 4px 0; font-size: 13px; color: #555;">
            Facility Coverage Progress ({coverage_pct}%)
        </div>
        <div style="background:#e0e0e0; border-radius:8px; height:18px; width:100%;">
            <div style="background:{bar_color}; width:{bar_pct*100:.1f}%;
                        height:18px; border-radius:8px; transition: width 0.5s;">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status banner
    banner_bg = {"green": "#d4edda", "orange": "#fff3cd", "red": "#f8d7da"}
    banner_border = {"green": "#28a745", "orange": "#ffc107", "red": "#dc3545"}
    color_key = req["status_color"]

    if gap < 0:
        gap_msg = (
            f"This district needs <b>{abs(gap)} more {req['amenity'].lower()}</b> "
            f"to meet the standard of 1 per {per_pop:,} residents."
        )
    else:
        gap_msg = (
            f"This district has a <b>surplus of {gap} {req['amenity'].lower()}</b> "
            f"relative to its population. Distribution equity may still vary by neighbourhood."
        )

    st.markdown(
        f"""
        <div style="margin-top:14px; padding:12px 16px; border-radius:8px;
                    background:{banner_bg[color_key]}; border-left: 5px solid {banner_border[color_key]};">
            <b>{status}</b> — {gap_msg}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Methodology note
    st.caption(
        f"💡 Standards based on MoHFW India, UDPFI Guidelines & Delhi Master Plan 2041. "
        f"OSM data may undercount private/informal facilities. "
        f"Population: Delhi Economic Survey 2024 district estimates."
    )


# 2. Sidebar
st.sidebar.header("Audit Settings")
delhi_districts = [
    "Central Delhi", "East Delhi", "New Delhi", "North Delhi",
    "North East Delhi", "North West Delhi", "Shahdara", "South Delhi",
    "South East Delhi", "South West Delhi", "West Delhi", "Old Delhi",
    "Central North Delhi", "Outer North Delhi",
]

district = st.sidebar.selectbox("Select Delhi District", delhi_districts)
amenity = st.sidebar.selectbox(
    "Essential Service", ["hospital", "school", "supermarket", "pharmacy"]
)

if st.sidebar.button("Clear Chat Memory"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a Delhi urban planning expert."}
    ]
    st.sidebar.success("Chat history cleared!")

# 3. Tabs
tab1, tab2, tab3 = st.tabs(["Live Audit", "Benchmarking", "Urban Planning Chat"])

with tab1:
    if st.sidebar.button("Run Audit"):
        with st.spinner(f"Analyzing {district}..."):
            full_place_name = f"{district}, Delhi, India"

            G = cached_network(full_place_name)
            pois = cached_pois(full_place_name, amenity)

            if pois.empty:
                st.error(f"No '{amenity}' found in {district}.")
                st.stop()

            target_nodes = ox.distance.nearest_nodes(G, pois.geometry.x, pois.geometry.y)
            distances = nx.multi_source_dijkstra_path_length(
                G, set(target_nodes), weight="time"
            )

            avg_time = sum(distances.values()) / len(distances)
            percent_served = (
                sum(1 for t in distances.values() if t <= 15) / len(distances)
            ) * 100

            display_df = pois[["name", "geometry"]].copy()
            display_df["Latitude"] = display_df.geometry.y
            display_df["Longitude"] = display_df.geometry.x
            display_df = display_df.drop(columns=["geometry"]).fillna("Unnamed Location")

            heatmap_data = [
                [G.nodes[n]["y"], G.nodes[n]["x"], max(0, 15 - t)]
                for n, t in distances.items()
                if t <= 20
            ]
            if len(heatmap_data) > 2000:
                heatmap_data = random.sample(heatmap_data, 2000)

            vals = list(distances.values())
            counts, bins = np.histogram(vals, bins=30, range=(0, 30))
            hist_df = pd.DataFrame({"Minutes": bins[:-1].round(1), "Nodes": counts})

            map_center = [pois.geometry.y.iloc[0], pois.geometry.x.iloc[0]]
            poi_coords = list(zip(
                pois.geometry.y.tolist(),
                pois.geometry.x.tolist(),
                pois["name"].fillna("Unnamed Location").tolist(),
            ))

            # ── Population requirement calculation ──────────────────
            req_data = calculate_facility_requirements(district, amenity, len(pois))
            # ────────────────────────────────────────────────────────

            audit_summary = (
                f"{district} {amenity}: Avg walk {avg_time:.1f}m, "
                f"15-min access {percent_served:.1f}%. "
                f"Required: {req_data['required']}, Found: {req_data['actual']}."
            )
            st.session_state.messages[0] = {
                "role": "system",
                "content": f"You are a Delhi expert. Context: {audit_summary}",
            }

            st.session_state.audit_results = {
                "avg_time": avg_time,
                "percent_served": percent_served,
                "display_df": display_df,
                "heatmap_data": heatmap_data,
                "hist_df": hist_df,
                "map_center": map_center,
                "poi_coords": poi_coords,
                "district": district,
                "amenity": amenity,
                "req_data": req_data,               # ← NEW
            }

            st.session_state.comparison_data.append({
                "District": district,
                "Service": amenity.title(),
                "Avg Walk (Min)": round(avg_time, 2),
                "15-Min Access %": round(percent_served, 2),
                "Required": req_data["required"],   # ← NEW
                "Actual (OSM)": req_data["actual"],       # ← NEW
                "Gap": req_data["gap"],              # ← NEW
            })

    if (
        st.session_state.audit_results is not None
        and "map_center" not in st.session_state.audit_results
    ):
        st.session_state.audit_results = None
        st.warning("Audit data was from an older session - please re-run the audit.")

    if st.session_state.audit_results:
        res = st.session_state.audit_results

        # ── 15-min metrics ──────────────────────────────────────────
        col1, col2 = st.columns(2)
        col1.metric("Avg. Walk Time", f"{res['avg_time']:.1f} mins")
        col2.metric("15-Min Access %", f"{res['percent_served']:.1f}%")

        if st.button("Get AI Insight"):
            with st.spinner("Generating insight..."):
                insight = get_ai_insight(
                    res["district"], res["amenity"],
                    res["avg_time"], res["percent_served"],
                )
                st.info(insight)

        # ── Population requirements panel ───────────────────────────
        if res.get("req_data"):
            render_requirements_panel(res["req_data"])

        st.markdown("---")

        # ── Map ─────────────────────────────────────────────────────
        m = folium.Map(location=res["map_center"], zoom_start=14)
        HeatMap(res["heatmap_data"]).add_to(m)

        for lat, lon, name in res["poi_coords"]:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#1565C0",
                fill=True,
                fill_color="#1E88E5",
                fill_opacity=0.85,
                weight=1.5,
                tooltip=name,
            ).add_to(m)

        st_folium(m, width=1100, height=500, returned_objects=[], key="audit_map")

        st.subheader("Walk Time Distribution")
        st.bar_chart(res["hist_df"].set_index("Minutes"), color="#2ecc71")

        st.dataframe(res["display_df"], width="stretch")
    else:
        st.info("Run an Audit from the sidebar to see spatial data.")

with tab2:
    if st.session_state.comparison_data:
        df_comp = pd.DataFrame(st.session_state.comparison_data).drop_duplicates()
        st.table(df_comp)
        st.bar_chart(df_comp.set_index("District")["15-Min Access %"])

with tab3:
    st.header("Urban Planning Assistant")
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
                stream=True,
            )
            full_response = ""
            placeholder = st.empty()
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + " ")
            placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})