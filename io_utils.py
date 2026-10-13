"""Vector upload parsing and geometry utilities."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.geometry import mapping


SUPPORTED_SUFFIXES = {".zip", ".geojson", ".json", ".gpkg"}


def _repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    try:
        out.geometry = out.geometry.make_valid()
    except Exception:
        out.geometry = out.geometry.buffer(0)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    return out


def load_vector_bytes(filename: str, payload: bytes) -> gpd.GeoDataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Upload a ZIP shapefile, GeoJSON/JSON, or GeoPackage.")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        if suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                safe_members = []
                for info in zf.infolist():
                    name = Path(info.filename)
                    if info.is_dir():
                        continue
                    if name.is_absolute() or ".." in name.parts:
                        raise ValueError("Unsafe path found inside uploaded ZIP.")
                    safe_members.append(info)
                for info in safe_members:
                    target = root / Path(info.filename).name
                    with zf.open(info) as src, open(target, "wb") as dst:
                        dst.write(src.read())

            shp_files = sorted(root.glob("*.shp"))
            if len(shp_files) == 0:
                raise ValueError("No .shp file was found in the ZIP archive.")
            if len(shp_files) > 1:
                raise ValueError("The ZIP contains more than one shapefile. Upload one shapefile per ZIP.")
            gdf = gpd.read_file(shp_files[0])

        else:
            local_path = root / Path(filename).name
            local_path.write_bytes(payload)
            gdf = gpd.read_file(local_path)

    if gdf.empty:
        raise ValueError("The uploaded vector file contains no features.")
    if gdf.crs is None:
        raise ValueError("The uploaded vector has no CRS. Define its projection before uploading it.")

    gdf = _repair_geometries(gdf)
    allowed = {"Polygon", "MultiPolygon"}
    bad = sorted(set(gdf.geometry.geom_type) - allowed)
    if bad:
        raise ValueError(f"Only polygon lake boundaries are supported. Found: {', '.join(bad)}")

    gdf = gdf.to_crs("EPSG:4326").reset_index(drop=True)
    gdf["_app_feature_id"] = [f"Feature {i+1}" for i in range(len(gdf))]
    return gdf


def geometry_for_selection(gdf: gpd.GeoDataFrame, selection: str):
    if selection == "All polygons (union)":
        try:
            geom = gdf.geometry.union_all()
        except AttributeError:
            geom = gdf.geometry.unary_union
        label = "All polygons"
    else:
        row = gdf[gdf["_app_feature_id"] == selection]
        if row.empty:
            raise ValueError(f"Could not find {selection}.")
        geom = row.geometry.iloc[0]
        label = selection
    return geom, label


def geometry_geojson(geom) -> dict:
    return {"type": "Feature", "properties": {}, "geometry": mapping(geom)}


def estimate_area_km2(geom) -> float:
    g = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("EPSG:6933")
    return float(g.area.iloc[0] / 1e6)
