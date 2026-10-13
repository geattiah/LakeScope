"""Google Earth Engine processing for the Lake Spectral Explorer."""

from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

import ee
import pandas as pd

from water_indices import INDEX_DEFINITIONS

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUDS = "COPERNICUS/S2_CLOUD_PROBABILITY"
JRC_WATER = "JRC/GSW1_4/GlobalSurfaceWater"


def initialize_ee(project: str, secrets=None) -> None:
    """Initialize EE locally or with Streamlit service-account secrets."""
    secrets = secrets or {}

    service_account = None
    private_key = None
    try:
        earth = secrets.get("earth_engine", {})
        service_account = earth.get("service_account")
        private_key = earth.get("private_key")
    except Exception:
        pass

    if service_account and private_key:
        credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
        ee.Initialize(credentials, project=project)
        return

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine is not authenticated. For local use, run `earthengine authenticate`; "
            "for Streamlit deployment, configure the service-account secrets described in README.md."
        ) from exc


def shapely_to_ee(geom):
    from shapely.geometry import mapping
    return ee.Geometry(mapping(geom))


def _mask_edges(image):
    return image.updateMask(image.select("B8A").mask().updateMask(image.select("B9").mask()))


def _mask_clouds_scale(image, cloud_probability_threshold: int):
    cloud_image = ee.Image(image.get("cloud_mask"))
    probability_mask = cloud_image.select("probability").lt(cloud_probability_threshold)
    scl = image.select("SCL")
    scl_mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    clean = probability_mask.And(scl_mask)
    optical = (
        image.select(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
        .divide(10000.0)
        .updateMask(clean)
    )
    return optical.copyProperties(
        image,
        ["system:time_start", "system:index", "CLOUDY_PIXEL_PERCENTAGE"],
    ).set("date_str", ee.Date(image.get("system:time_start")).format("YYYY-MM-dd"))


def build_collection(aoi, start_date, end_date, max_scene_cloud=80, cloud_probability_threshold=40):
    # Earth Engine filterDate end is exclusive; advance user's inclusive end date by one day.
    end_exclusive = ee.Date(str(end_date)).advance(1, "day")
    criteria = ee.Filter.And(
        ee.Filter.bounds(aoi),
        ee.Filter.date(str(start_date), end_exclusive),
    )
    sr = (
        ee.ImageCollection(S2_SR)
        .filter(criteria)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_scene_cloud))
        .map(_mask_edges)
    )
    clouds = ee.ImageCollection(S2_CLOUDS).filter(criteria)
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=sr,
        secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return (
        ee.ImageCollection(joined)
        .filter(ee.Filter.notNull(["cloud_mask"]))
        .map(lambda img: _mask_clouds_scale(ee.Image(img), cloud_probability_threshold))
    )


def persistent_water_mask(water_occurrence_threshold: int):
    return ee.Image(JRC_WATER).select("occurrence").gte(water_occurrence_threshold)


def calculate_index(image, index_key: str, water_occurrence_threshold: int):
    image = ee.Image(image)
    b2 = image.select("B2")
    b3 = image.select("B3")
    b4 = image.select("B4")
    b5 = image.select("B5")
    b6 = image.select("B6")
    b8a = image.select("B8A")
    eps = ee.Image.constant(1e-6)

    if index_key == "NDCI":
        denominator = b5.add(b4).abs().max(eps)
        idx = b5.subtract(b4).divide(denominator)
    elif index_key == "MCI":
        idx = b5.subtract(b4.add(b6.subtract(b4).multiply((705.0 - 665.0) / (740.0 - 665.0))))
    elif index_key == "GREEN_BLUE_RATIO":
        idx = b3.divide(b2.abs().max(eps))
    elif index_key == "NGBI":
        denominator = b3.add(b2).abs().max(eps)
        idx = b3.subtract(b2).divide(denominator)
    elif index_key == "NDTI":
        denominator = b4.add(b3).abs().max(eps)
        idx = b4.subtract(b3).divide(denominator)
    elif index_key == "RED_NIR_RATIO":
        idx = b4.divide(b8a.abs().max(eps))
    else:
        raise ValueError(f"Unknown index: {index_key}")

    return (
        idx.rename("index")
        .updateMask(persistent_water_mask(water_occurrence_threshold))
        .copyProperties(image, ["system:time_start", "date_str"])
    )


def acquisition_dates(collection) -> List[str]:
    dates = ee.List(collection.aggregate_array("date_str")).distinct().sort().getInfo()
    return [str(x) for x in dates]


def daily_index_image(collection, date_string: str, index_key: str, water_occurrence_threshold: int, aoi):
    start = ee.Date(date_string)
    day = collection.filterDate(start, start.advance(1, "day"))
    mosaic = day.sort("CLOUDY_PIXEL_PERCENTAGE", False).mosaic().clip(aoi)
    return calculate_index(mosaic, index_key, water_occurrence_threshold).clip(aoi)


def period_index_image(collection, index_key: str, water_occurrence_threshold: int, aoi, reducer: str):
    indexed = collection.map(lambda img: calculate_index(ee.Image(img), index_key, water_occurrence_threshold))
    if reducer == "Median":
        out = indexed.median()
    elif reducer == "P90":
        out = indexed.reduce(ee.Reducer.percentile([90])).rename("index")
    elif reducer == "Mean":
        out = indexed.mean()
    else:
        raise ValueError(reducer)
    return ee.Image(out).rename("index").clip(aoi)


def _stat_reducer():
    return (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.median(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.percentile([90]), sharedInputs=True)
        .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
    )


def temporal_stats_acquisitions(
    collection,
    aoi,
    index_key: str,
    water_occurrence_threshold: int,
    scale: int,
) -> pd.DataFrame:
    dates = ee.List(collection.aggregate_array("date_str")).distinct().sort()
    water = persistent_water_mask(water_occurrence_threshold)
    total_area_dict = ee.Image.pixelArea().updateMask(water).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
        maxPixels=1e10, bestEffort=True, tileScale=4,
    )
    total_area = ee.Number(ee.Algorithms.If(
        total_area_dict.contains("area"), total_area_dict.get("area"), 0
    ))

    def one(ds):
        ds = ee.String(ds)
        start = ee.Date(ds)
        day = collection.filterDate(start, start.advance(1, "day"))
        mosaic = day.sort("CLOUDY_PIXEL_PERCENTAGE", False).mosaic().clip(aoi)
        idx = calculate_index(mosaic, index_key, water_occurrence_threshold).clip(aoi)
        stats = idx.reduceRegion(
            reducer=_stat_reducer(), geometry=aoi, scale=scale,
            maxPixels=1e10, bestEffort=True, tileScale=4,
        )
        valid_area_dict = ee.Image.pixelArea().updateMask(idx.mask()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e10, bestEffort=True, tileScale=4,
        )
        valid_area = ee.Number(ee.Algorithms.If(
            valid_area_dict.contains("area"), valid_area_dict.get("area"), 0
        ))
        valid_fraction = ee.Algorithms.If(total_area.gt(0), valid_area.divide(total_area), 0)
        props = ee.Dictionary(stats).combine({
            "date": ds,
            "source_scenes": day.size(),
            "valid_fraction": valid_fraction,
        })
        return ee.Feature(None, props)

    fc = ee.FeatureCollection(dates.map(one))
    info = fc.getInfo()
    rows = [f.get("properties", {}) for f in info.get("features", [])]
    return _clean_stats_df(rows)


def temporal_stats_monthly(
    collection,
    aoi,
    start_date,
    end_date,
    index_key: str,
    water_occurrence_threshold: int,
    scale: int,
) -> pd.DataFrame:
    start = ee.Date(str(start_date)).update(day=1)
    end = ee.Date(str(end_date)).advance(1, "day")
    n_months = end.difference(start, "month").ceil()
    offsets = ee.List.sequence(0, n_months.subtract(1))
    indexed = collection.map(lambda img: calculate_index(ee.Image(img), index_key, water_occurrence_threshold))
    water = persistent_water_mask(water_occurrence_threshold)
    total_area_dict = ee.Image.pixelArea().updateMask(water).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
        maxPixels=1e10, bestEffort=True, tileScale=4,
    )
    total_area = ee.Number(ee.Algorithms.If(
        total_area_dict.contains("area"), total_area_dict.get("area"), 0
    ))

    def one(offset):
        ms = start.advance(ee.Number(offset), "month")
        me = ms.advance(1, "month")
        source = collection.filterDate(ms, me)
        month_idx = indexed.filterDate(ms, me).median().rename("index").clip(aoi)
        stats = month_idx.reduceRegion(
            reducer=_stat_reducer(), geometry=aoi, scale=scale,
            maxPixels=1e10, bestEffort=True, tileScale=4,
        )
        valid_area_dict = ee.Image.pixelArea().updateMask(month_idx.mask()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e10, bestEffort=True, tileScale=4,
        )
        valid_area = ee.Number(ee.Algorithms.If(
            valid_area_dict.contains("area"), valid_area_dict.get("area"), 0
        ))
        valid_fraction = ee.Algorithms.If(total_area.gt(0), valid_area.divide(total_area), 0)
        props = ee.Dictionary(stats).combine({
            "date": ms.format("YYYY-MM-01"),
            "source_scenes": source.size(),
            "valid_fraction": valid_fraction,
        })
        return ee.Feature(None, props)

    fc = ee.FeatureCollection(offsets.map(one)).filter(ee.Filter.gt("source_scenes", 0))
    info = fc.getInfo()
    rows = [f.get("properties", {}) for f in info.get("features", [])]
    return _clean_stats_df(rows)


def _clean_stats_df(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "mean", "median", "p90", "max", "valid_fraction", "source_scenes"])
    df = pd.DataFrame(rows)
    rename = {
        "index_mean": "mean",
        "index_median": "median",
        "index_p90": "p90",
        "index_max": "max",
    }
    df = df.rename(columns=rename)
    for col in ["mean", "median", "p90", "max", "valid_fraction", "source_scenes"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df[["date", "mean", "median", "p90", "max", "valid_fraction", "source_scenes"]].sort_values("date")


def vis_params(index_key: str, vmin: Optional[float] = None, vmax: Optional[float] = None):
    spec = INDEX_DEFINITIONS[index_key]
    return {
        "min": spec["vis_min"] if vmin is None else vmin,
        "max": spec["vis_max"] if vmax is None else vmax,
        "palette": spec["palette"],
    }
