# Lake Spectral Explorer — MVP

A Streamlit + Google Earth Engine application for spatial and temporal Sentinel-2 spectral-index analysis over user-uploaded lake polygons.

## Included indices

**Chlorophyll / algal proxy**
- NDCI: `(B5-B4)/(B5+B4)`
- MCI: red-edge maximum chlorophyll index using B4/B5/B6

**DOC / CDOM proxy**
- Green/Blue ratio: `B3/B2`
- Normalized Green–Blue Index: `(B3-B2)/(B3+B2)`

**Turbidity proxy**
- NDTI: `(B4-B3)/(B4+B3)`
- Red/NIR ratio: `B4/B8A`

These are spectral proxies, not calibrated concentrations.

## MVP workflow

1. Upload a zipped shapefile, GeoJSON/JSON, or GeoPackage.
2. The app validates polygon geometry and reprojects it to WGS84.
3. Choose one polygon or analyze the union of all polygons.
4. Choose water-quality property and index.
5. Choose Sentinel-2 date range and QC thresholds.
6. Run either every-acquisition or monthly-median temporal analysis.
7. View mean/median/P90/max index time series and valid-water coverage.
8. View a spatial index map for a selected acquisition or a period composite.
9. Download the temporal CSV and an on-demand GeoTIFF.

## Local run

Create/activate your Python environment, then:

```bash
pip install -r requirements.txt
earthengine authenticate
streamlit run streamlit_app.py
```

The Earth Engine project is currently set in `streamlit_app.py` to:

```text
gg-wa-temp-2025
```

## Streamlit deployment

For hosted deployment, use an Earth Engine-authorized Google Cloud service account. Copy the structure from `.streamlit/secrets.toml.example` into the Streamlit app's Secrets settings. Do not commit a real private key to GitHub.

## Current MVP limits

- 50 polygon features per upload.
- Interactive map uses Earth Engine tile URLs during the active Streamlit session.
- GeoTIFF download is intended for lake-sized regions; very large AOIs can exceed Earth Engine download limits.
- No concentration calibration is applied.
- No trend significance test or anomaly climatology is included yet.

## Next build targets

- Multi-lake comparison charts using a lake-name field.
- Seasonal/anomaly products and long-term trend metrics.
- Better acquisition browser with cloud/coverage metadata.
- Downloadable PNG figures and packaged ZIP reports.
- Optional RGB viewer separate from index viewer.
- Saved project/session configuration.
