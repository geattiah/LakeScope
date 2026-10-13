"""Sentinel-2 spectral indices used by the Lake Spectral Explorer."""

INDEX_DEFINITIONS = {
    "NDCI": {
        "group": "Chlorophyll / algal proxy",
        "label": "NDCI — Normalized Difference Chlorophyll Index",
        "formula": "(B5 - B4) / (B5 + B4)",
        "units": "unitless",
        "scale": 20,
        "vis_min": -0.30,
        "vis_max": 0.50,
        "palette": ["440154", "3b528b", "21918c", "5ec962", "fde725", "fdae61", "d73027"],
        "description": "Red/red-edge chlorophyll and algal-biomass proxy for inland waters.",
    },
    "MCI": {
        "group": "Chlorophyll / algal proxy",
        "label": "MCI — Maximum Chlorophyll Index",
        "formula": "B5 - [B4 + (B6-B4) × (705-665)/(740-665)]",
        "units": "reflectance",
        "scale": 20,
        "vis_min": -0.03,
        "vis_max": 0.05,
        "palette": ["313695", "74add1", "ffffbf", "f46d43", "a50026"],
        "description": "Red-edge peak proxy associated with elevated chlorophyll/algal biomass.",
    },
    "GREEN_BLUE_RATIO": {
        "group": "DOC / CDOM proxy",
        "label": "Green/Blue ratio — B3/B2",
        "formula": "B3 / B2",
        "units": "ratio",
        "scale": 20,
        "vis_min": 0.5,
        "vis_max": 2.5,
        "palette": ["313695", "74add1", "ffffbf", "fdae61", "a50026"],
        "description": "Optical proxy for coloured dissolved organic matter (CDOM), not direct DOC concentration.",
    },
    "NGBI": {
        "group": "DOC / CDOM proxy",
        "label": "Normalized Green–Blue Index",
        "formula": "(B3 - B2) / (B3 + B2)",
        "units": "unitless",
        "scale": 20,
        "vis_min": -0.30,
        "vis_max": 0.40,
        "palette": ["313695", "74add1", "ffffbf", "fdae61", "a50026"],
        "description": "Normalized blue–green water-colour proxy related to CDOM/DOC variability.",
    },
    "NDTI": {
        "group": "Turbidity proxy",
        "label": "NDTI — Normalized Difference Turbidity Index",
        "formula": "(B4 - B3) / (B4 + B3)",
        "units": "unitless",
        "scale": 20,
        "vis_min": -0.30,
        "vis_max": 0.40,
        "palette": ["313695", "74add1", "ffffbf", "fdae61", "a50026"],
        "description": "Red–green optical proxy for suspended sediment/turbidity.",
    },
    "RED_NIR_RATIO": {
        "group": "Turbidity proxy",
        "label": "Red/NIR ratio — B4/B8A",
        "formula": "B4 / B8A",
        "units": "ratio",
        "scale": 20,
        "vis_min": 0.0,
        "vis_max": 4.0,
        "palette": ["313695", "74add1", "ffffbf", "fdae61", "a50026"],
        "description": "Complementary red/NIR suspended-material proxy; interpret comparatively rather than as NTU.",
    },
}


def keys_for_group(group):
    return [k for k, v in INDEX_DEFINITIONS.items() if v["group"] == group]
