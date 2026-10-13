from __future__ import annotations

import hashlib
import tempfile
from datetime import date
from pathlib import Path

import ee
import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from gee_core import (
    acquisition_dates,
    build_collection,
    daily_index_image,
    initialize_ee,
    period_index_image,
    shapely_to_ee,
    temporal_stats_acquisitions,
    temporal_stats_monthly,
    vis_params,
)
from io_utils import estimate_area_km2, geometry_for_selection, geometry_geojson, load_vector_bytes
from water_indices import INDEX_DEFINITIONS, keys_for_group


st.set_page_config(page_title="Lake Spectral Explorer", layout="wide")

GEE_PROJECT = "gg-wa-temp-2025"
MIN_S2_DATE = date(2017, 3, 28)
MAX_UPLOAD_FEATURES = 50


@st.cache_data(show_spinner=False)
def parse_upload(filename: str, payload: bytes):
    return load_vector_bytes(filename, payload)


def add_ee_layer(m, image, params, name):
    map_id = ee.Image(image).getMapId(params)
    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
        show=True,
    ).add_to(m)


def map_for_image(image, geom_geojson, bounds, params, title):
    minx, miny, maxx, maxy = bounds
    m = folium.Map(tiles=None, control_scale=True)
    add_ee_layer(m, image, params, title)
    folium.GeoJson(
        geom_geojson,
        name="Uploaded boundary",
        style_function=lambda _: {"color": "black", "weight": 2.5, "fillOpacity": 0},
    ).add_to(m)
    cmap = LinearColormap(
        colors=[f"#{c}" for c in params["palette"]],
        vmin=params["min"], vmax=params["max"], caption=title,
    )
    cmap.add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def current_analysis_key(file_bytes, selection, index_key, start_date, end_date, scene_cloud, cloud_prob, water_occ, temporal_mode):
    h = hashlib.sha256(file_bytes).hexdigest()[:16]
    return "|".join(map(str, [h, selection, index_key, start_date, end_date, scene_cloud, cloud_prob, water_occ, temporal_mode]))


st.title("Lake Spectral Explorer")
st.caption(
    "Upload lake polygons, calculate Sentinel-2 water-quality spectral indices in Google Earth Engine, "
    "inspect spatial patterns, and examine temporal change."
)

with st.expander("What the indices represent", expanded=False):
    st.write(
        "These outputs are spectral indices/proxies, not calibrated concentrations. "
        "NDCI and MCI are chlorophyll/algal proxies; the blue–green indices are CDOM/DOC proxies; "
        "and NDTI/red–NIR are turbidity/suspended-material proxies."
    )

try:
    initialize_ee(GEE_PROJECT, st.secrets)
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded = st.file_uploader(
    "Upload lake polygons",
    type=["zip", "geojson", "json", "gpkg"],
    help="ZIP shapefiles must include .shp, .shx, .dbf and .prj files.",
)

if uploaded is None:
    st.info("Upload a lake polygon file to begin.")
    st.stop()

payload = uploaded.getvalue()
try:
    gdf = parse_upload(uploaded.name, payload)
except Exception as exc:
    st.error(f"Could not read the uploaded vector: {exc}")
    st.stop()

if len(gdf) > MAX_UPLOAD_FEATURES:
    st.error(f"This MVP currently supports up to {MAX_UPLOAD_FEATURES} polygon features per upload.")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.metric("Polygon features", len(gdf))
with right:
    st.metric("Input CRS after validation", "EPSG:4326")

selection_options = ["All polygons (union)"] + gdf["_app_feature_id"].tolist()
selection = st.selectbox("Analysis polygon", selection_options)
geom, geom_label = geometry_for_selection(gdf, selection)
area_km2 = estimate_area_km2(geom)
st.caption(f"Selected area: approximately {area_km2:,.2f} km²")

if area_km2 > 10000:
    st.warning("This is a large analysis area. Sentinel-2 reductions and downloads may be slow.")

# Configuration form
with st.form("analysis_controls"):
    c1, c2, c3 = st.columns(3)
    with c1:
        group = st.selectbox(
            "Water-quality property",
            ["Chlorophyll / algal proxy", "DOC / CDOM proxy", "Turbidity proxy"],
        )
        index_key = st.selectbox(
            "Spectral index",
            keys_for_group(group),
            format_func=lambda k: INDEX_DEFINITIONS[k]["label"],
        )
        spec = INDEX_DEFINITIONS[index_key]
        st.caption(f"Formula: {spec['formula']}")

    with c2:
        start_date = st.date_input("Start date", value=date(2024, 6, 1), min_value=MIN_S2_DATE, max_value=date.today())
        end_date = st.date_input("End date", value=date.today(), min_value=MIN_S2_DATE, max_value=date.today())
        temporal_mode = st.selectbox("Temporal series", ["Every acquisition", "Monthly median"])

    with c3:
        scene_cloud = st.slider("Maximum scene cloud (%)", 0, 100, 80, 5)
        cloud_prob = st.slider("Maximum cloud probability (%)", 0, 100, 40, 5)
        water_occ = st.slider("Persistent-water occurrence threshold (%)", 0, 100, 50, 5)
        min_valid = st.slider("Minimum valid-water coverage (%)", 0, 100, 20, 5)

    run = st.form_submit_button("Run Sentinel-2 analysis", type="primary")

if start_date > end_date:
    st.error("The start date must be earlier than or equal to the end date.")
    st.stop()

analysis_key = current_analysis_key(payload, selection, index_key, start_date, end_date, scene_cloud, cloud_prob, water_occ, temporal_mode)

if run:
    ee_geom = shapely_to_ee(geom)
    with st.spinner("Searching Sentinel-2 and calculating the temporal series..."):
        collection = build_collection(
            ee_geom, start_date, end_date,
            max_scene_cloud=scene_cloud,
            cloud_probability_threshold=cloud_prob,
        )
        scene_count = int(collection.size().getInfo())
        if scene_count == 0:
            st.warning("No Sentinel-2 scenes were found for this polygon and date range.")
            st.stop()

        scale = INDEX_DEFINITIONS[index_key]["scale"]
        if temporal_mode == "Every acquisition":
            df = temporal_stats_acquisitions(collection, ee_geom, index_key, water_occ, scale)
        else:
            df = temporal_stats_monthly(collection, ee_geom, start_date, end_date, index_key, water_occ, scale)

        df["usable"] = (
            pd.to_numeric(df["valid_fraction"], errors="coerce").fillna(0) >= (min_valid / 100.0)
        ) & df["mean"].notna()

        st.session_state["analysis"] = {
            "key": analysis_key,
            "df": df,
            "scene_count": scene_count,
            "index_key": index_key,
            "selection": selection,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "scene_cloud": scene_cloud,
            "cloud_prob": cloud_prob,
            "water_occ": water_occ,
            "min_valid": min_valid,
            "temporal_mode": temporal_mode,
        }

analysis = st.session_state.get("analysis")
if not analysis or analysis.get("key") != analysis_key:
    st.info("Choose the controls above and click **Run Sentinel-2 analysis**.")
    st.stop()

results = analysis["df"].copy()
usable = results[results["usable"]].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Source S2 scenes", analysis["scene_count"])
m2.metric("Time steps", len(results))
m3.metric("Usable time steps", len(usable))
m4.metric("Usable fraction", f"{(len(usable)/len(results)*100 if len(results) else 0):.0f}%")

if usable.empty:
    st.warning("No time steps passed the selected valid-water coverage threshold. Lower the threshold or revise the date range/QC settings.")
    st.dataframe(results, use_container_width=True)
    st.stop()

st.subheader("Temporal change")
metric_label = st.selectbox("Statistic shown in graph", ["p90", "median", "mean", "max"], index=0)
chart_df = usable[["date", metric_label]].dropna().set_index("date")
st.line_chart(chart_df, y=metric_label)

with st.expander("Temporal/QC table", expanded=False):
    display_df = results.copy()
    display_df["valid_water_percent"] = pd.to_numeric(display_df["valid_fraction"], errors="coerce") * 100
    st.dataframe(display_df, use_container_width=True)

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download temporal CSV",
    data=csv_bytes,
    file_name=f"{geom_label.replace(' ', '_')}_{index_key}_temporal.csv",
    mime="text/csv",
)

st.subheader("Spatial index")
spatial_mode = st.radio("Spatial product", ["Single acquisition", "Period median", "Period P90", "Period mean"], horizontal=True)

# Rebuild lazy EE objects from stored settings.
ee_geom = shapely_to_ee(geom)
collection = build_collection(
    ee_geom,
    analysis["start_date"],
    analysis["end_date"],
    max_scene_cloud=analysis["scene_cloud"],
    cloud_probability_threshold=analysis["cloud_prob"],
)

if spatial_mode == "Single acquisition":
    date_options = usable["date"].dt.strftime("%Y-%m-%d").tolist()
    selected_date = st.selectbox("Usable acquisition date", date_options, index=len(date_options)-1)
    spatial_image = daily_index_image(collection, selected_date, index_key, water_occ, ee_geom)
    spatial_title = f"{INDEX_DEFINITIONS[index_key]['label']} — {selected_date}"
else:
    reducer = spatial_mode.replace("Period ", "")
    spatial_image = period_index_image(collection, index_key, water_occ, ee_geom, reducer)
    spatial_title = f"{INDEX_DEFINITIONS[index_key]['label']} — {spatial_mode}"

spec = INDEX_DEFINITIONS[index_key]
vcol1, vcol2 = st.columns(2)
with vcol1:
    vmin = st.number_input("Map minimum", value=float(spec["vis_min"]), format="%.4f")
with vcol2:
    vmax = st.number_input("Map maximum", value=float(spec["vis_max"]), format="%.4f")
if vmin >= vmax:
    st.error("Map minimum must be less than map maximum.")
    st.stop()

params = vis_params(index_key, vmin, vmax)
bounds = geom.bounds
map_obj = map_for_image(spatial_image, geometry_geojson(geom), bounds, params, spatial_title)
st_folium(map_obj, use_container_width=True, height=620, returned_objects=[])

# Small, on-demand GeoTIFF download. geemap handles the Earth Engine transfer.
if st.button("Prepare GeoTIFF download"):
    import geemap
    with st.spinner("Preparing GeoTIFF..."):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.close()
            geemap.download_ee_image(
                image=spatial_image.toFloat(),
                filename=tmp.name,
                region=ee_geom,
                scale=spec["scale"],
            )
            tif_bytes = Path(tmp.name).read_bytes()
            st.session_state["tif_bytes"] = tif_bytes
            st.session_state["tif_name"] = f"{geom_label.replace(' ', '_')}_{index_key}_{spatial_mode.replace(' ', '_')}.tif"
        except Exception as exc:
            st.error(f"GeoTIFF preparation failed: {exc}")

if st.session_state.get("tif_bytes"):
    st.download_button(
        "Download GeoTIFF",
        data=st.session_state["tif_bytes"],
        file_name=st.session_state.get("tif_name", "lake_index.tif"),
        mime="image/tiff",
    )

st.caption(
    "Interpretation: these products are Sentinel-2 spectral indices/proxies. "
    "They should not be reported as chlorophyll, DOC, or turbidity concentrations without independent calibration."
)
