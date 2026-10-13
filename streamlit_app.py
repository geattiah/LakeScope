from __future__ import annotations

import hashlib
import tempfile
from datetime import date
from pathlib import Path

import ee
import folium
import pandas as pd
import streamlit as st
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


st.set_page_config(page_title="LakeScope", layout="wide")

GEE_PROJECT = "gg-wa-temp-2025"
MIN_S2_DATE = date(2017, 3, 28)
MAX_UPLOAD_FEATURES = 50


def ui_text(value) -> str:
    """Use plain hyphens in user-facing text."""
    return str(value).replace("-", "-").replace("-", "-")


def preview_uploaded_geometry(geom_geojson, bounds, label):
    """Build a lightweight upload-preview map with no Earth Engine work."""
    minx, miny, maxx, maxy = bounds
    center = [(miny + maxy) / 2.0, (minx + maxx) / 2.0]
    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    folium.GeoJson(
        geom_geojson,
        name=ui_text(label),
        style_function=lambda _: {
            "color": "#111111",
            "weight": 3,
            "fillColor": "#4C78A8",
            "fillOpacity": 0.18,
        },
    ).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    return m


def add_clean_legend(m, params, title):
    """Add a compact legend with labels separated from the color bar."""
    palette = ["#" + str(c).lstrip("#") for c in params["palette"]]
    gradient = ", ".join(palette)
    vmin = float(params["min"])
    vmax = float(params["max"])
    safe_title = ui_text(title)
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 28px; right: 22px; z-index: 9999;
        width: min(320px, calc(100vw - 44px));
        background: rgba(255,255,255,0.96);
        border: 1px solid #999; border-radius: 6px;
        padding: 10px 12px 9px 12px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif; font-size: 12px; color: #111;
    "">
      <div style="font-weight:600; line-height:1.25; margin-bottom:7px; word-break:break-word;">{safe_title}</div>
      <div style="height:14px; border:1px solid #777; background:linear-gradient(to right, {gradient});"></div>
      <div style="display:flex; justify-content:space-between; margin-top:4px; line-height:1;">
        <span>{vmin:g}</span><span>{vmax:g}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


@st.cache_data(show_spinner=False)
def parse_upload(filename: str, payload: bytes):
    return load_vector_bytes(filename, payload)


def download_ee_geotiff(image, ee_geom, scale: int) -> bytes:
    """Download one EE image to an in-memory GeoTIFF for mapping/download.

    This intentionally avoids ``Image.getMapId()``. Hosted Streamlit service
    accounts can reach the EE computation API while map-tile creation can fail
    separately. A real GeoTIFF is also preferable because the map and the user
    download are derived from the same spatial product.
    """
    import geemap

    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        geemap.download_ee_image(
            image=ee.Image(image).toFloat(),
            filename=str(tmp_path),
            region=ee_geom,
            scale=scale,
        )
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError("Earth Engine returned an empty GeoTIFF.")
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def geotiff_to_rgba(tif_bytes: bytes, params):
    """Reproject a single-band GeoTIFF to WGS84 and colour it for Folium.

    Reprojecting the raster pixels themselves (rather than merely transforming
    the bounding box) prevents raster/vector shifts in the browser map.
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from rasterio.io import MemoryFile
    from rasterio.transform import array_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            if src.count < 1:
                raise RuntimeError("Downloaded GeoTIFF has no raster band.")
            if src.crs is None:
                raise RuntimeError("Downloaded GeoTIFF has no CRS.")

            source = src.read(1, masked=True).astype("float32").filled(np.nan)
            dst_crs = "EPSG:4326"

            if str(src.crs).upper() == dst_crs:
                data = source
                transform = src.transform
                height, width = src.height, src.width
            else:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                data = np.full((height, width), np.nan, dtype="float32")
                reproject(
                    source=source,
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                )

    valid = np.isfinite(data)
    vmin = float(params["min"])
    vmax = float(params["max"])
    norm = np.zeros_like(data, dtype="float32")
    norm[valid] = np.clip((data[valid] - vmin) / (vmax - vmin), 0, 1)

    colors = ["#" + str(c).lstrip("#") for c in params["palette"]]
    cmap = LinearSegmentedColormap.from_list("lakescope_index", colors)
    rgba = np.round(cmap(norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 235, 0).astype("uint8")

    west, south, east, north = array_bounds(height, width, transform)
    leaflet_bounds = [[south, west], [north, east]]
    return rgba, leaflet_bounds


def map_for_geotiff(tif_bytes, geom_geojson, bounds, params, title):
    minx, miny, maxx, maxy = bounds
    rgba, raster_bounds = geotiff_to_rgba(tif_bytes, params)

    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
    m.get_root().header.add_child(
        folium.Element("<style>.leaflet-container{background:white !important;}</style>")
    )
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=raster_bounds,
        name=title,
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=2,
        show=True,
    ).add_to(m)
    folium.GeoJson(
        geom_geojson,
        name="Uploaded boundary",
        style_function=lambda _: {"color": "black", "weight": 2.5, "fillOpacity": 0},
    ).add_to(m)
    add_clean_legend(m, params, title)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def current_analysis_key(file_bytes, selection, index_key, start_date, end_date, scene_cloud, cloud_prob, water_occ, temporal_mode):
    h = hashlib.sha256(file_bytes).hexdigest()[:16]
    return "|".join(map(str, [h, selection, index_key, start_date, end_date, scene_cloud, cloud_prob, water_occ, temporal_mode]))


st.title("LakeScope")
st.caption(
    "Upload lake polygons, calculate Sentinel-2 water-quality spectral indices in Google Earth Engine, "
    "inspect spatial patterns, and examine temporal change."
)

with st.expander("What the indices represent", expanded=False):
    st.write(
        "These outputs are spectral indices/proxies, not calibrated concentrations. "
        "NDCI and MCI are chlorophyll/algal proxies; the blue-green indices are CDOM/DOC proxies; "
        "and NDTI/red-NIR are turbidity/suspended-material proxies."
    )

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

st.subheader("Uploaded lake boundary")
st.caption("Preview of the selected polygon before any Sentinel-2 processing.")
preview_map = preview_uploaded_geometry(geometry_geojson(geom), geom.bounds, geom_label)
st_folium(
    preview_map,
    use_container_width=True,
    height=430,
    returned_objects=[],
    key=f"upload_preview_{hashlib.sha256(payload).hexdigest()[:12]}_{selection}",
)

# Reactive property/index controls.
# These intentionally live OUTSIDE the form so changing the water-quality
# property immediately refreshes the available indices and their description.
group = st.selectbox(
    "Water-quality property",
    ["Chlorophyll / algal proxy", "DOC / CDOM proxy", "Turbidity proxy"],
    key="water_quality_group",
)

index_options = keys_for_group(group)
index_key = st.selectbox(
    "Spectral index",
    index_options,
    format_func=lambda k: ui_text(INDEX_DEFINITIONS[k]["label"]),
    # A group-specific key makes Streamlit select the first valid index when
    # the property changes (e.g., NDTI for Turbidity) instead of retaining a
    # stale index from the previous property.
    key=f"spectral_index_{group}",
)
spec = INDEX_DEFINITIONS[index_key]
st.caption(f"Formula: {spec['formula']}")
st.caption(ui_text(spec["description"]))

# Computational controls remain in a form so expensive analysis only runs
# when the user explicitly clicks the Run button.
with st.form("analysis_controls"):
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start date", value=date(2024, 6, 1), min_value=MIN_S2_DATE, max_value=date.today())
        end_date = st.date_input("End date", value=date.today(), min_value=MIN_S2_DATE, max_value=date.today())
        temporal_mode = st.selectbox("Temporal series", ["Every acquisition", "Monthly median"])

    with c2:
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
    try:
        initialize_ee(GEE_PROJECT, st.secrets)
    except Exception as exc:
        st.error(f"Earth Engine initialization failed: {exc}")
        st.stop()

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
try:
    initialize_ee(GEE_PROJECT, st.secrets)
except Exception as exc:
    st.error(f"Earth Engine initialization failed: {exc}")
    st.stop()

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
    spatial_title = f"{ui_text(INDEX_DEFINITIONS[index_key]['label'])} - {selected_date}"
else:
    reducer = spatial_mode.replace("Period ", "")
    spatial_image = period_index_image(collection, index_key, water_occ, ee_geom, reducer)
    spatial_title = f"{ui_text(INDEX_DEFINITIONS[index_key]['label'])} - {spatial_mode}"

spec = INDEX_DEFINITIONS[index_key]
vcol1, vcol2 = st.columns(2)
with vcol1:
    vmin = st.number_input(
        "Map minimum",
        value=float(spec["vis_min"]),
        format="%.4f",
        key=f"map_min_{index_key}",
    )
with vcol2:
    vmax = st.number_input(
        "Map maximum",
        value=float(spec["vis_max"]),
        format="%.4f",
        key=f"map_max_{index_key}",
    )
if vmin >= vmax:
    st.error("Map minimum must be less than map maximum.")
    st.stop()

params = vis_params(index_key, vmin, vmax)
bounds = geom.bounds

spatial_cache_key = "|".join(
    map(str, [
        analysis_key, spatial_mode,
        selected_date if spatial_mode == "Single acquisition" else "period",
        index_key, spec["scale"],
    ])
)

if st.session_state.get("spatial_tif_key") != spatial_cache_key:
    with st.spinner("Preparing the spatial index map..."):
        try:
            st.session_state["spatial_tif_bytes"] = download_ee_geotiff(
                spatial_image, ee_geom, spec["scale"]
            )
            st.session_state["spatial_tif_key"] = spatial_cache_key
            st.session_state["spatial_tif_name"] = (
                f"{geom_label.replace(' ', '_')}_{index_key}_"
                f"{(selected_date if spatial_mode == 'Single acquisition' else spatial_mode).replace(' ', '_')}.tif"
            )
        except Exception as exc:
            st.error(f"Spatial GeoTIFF preparation failed: {exc}")
            st.stop()

tif_bytes = st.session_state.get("spatial_tif_bytes")
if not tif_bytes:
    st.error("The spatial raster could not be prepared.")
    st.stop()

try:
    map_obj = map_for_geotiff(
        tif_bytes, geometry_geojson(geom), bounds, params, spatial_title
    )
    st_folium(map_obj, use_container_width=True, height=620, returned_objects=[])
except Exception as exc:
    st.error(f"Spatial map rendering failed: {exc}")
    st.stop()

st.download_button(
    "Download GeoTIFF",
    data=tif_bytes,
    file_name=st.session_state.get("spatial_tif_name", "lake_index.tif"),
    mime="image/tiff",
)

st.caption(
    "Interpretation: these products are Sentinel-2 spectral indices/proxies. "
    "They should not be reported as chlorophyll, DOC, or turbidity concentrations without independent calibration."
)
