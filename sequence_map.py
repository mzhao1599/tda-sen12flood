import os
import glob
import re
import sys
import geopandas as gpd
import folium
import pandas as pd
FOLDER = r"labels"  
OUTPUT_HTML = r"all_sequences_map.html"
INITIAL_ZOOM = 8
if not os.path.isdir(FOLDER):
    print(f"Error: folder not found: {FOLDER}")
    sys.exit(1)

all_files = glob.glob(os.path.join(FOLDER, "*.geojson"))
if not all_files:
    print(f"No .geojson files in {FOLDER}")
    sys.exit(1)
pat = re.compile(r"^[^_]+_([^_]+)_")  
seq_map = {}
for fp in all_files:
    name = os.path.basename(fp)
    m = pat.match(name)
    if not m:
        continue
    seq = m.group(1)
    if seq not in seq_map:
        seq_map[seq] = fp
selected = list(seq_map.values())
print(f"Found {len(selected)} unique sequences out of {len(all_files)} files.")
gdfs = [gpd.read_file(fp) for fp in selected]
df = pd.concat(gdfs, ignore_index=True)
combined = gpd.GeoDataFrame(df, geometry="geometry", crs=gdfs[0].crs)
if combined.crs and combined.crs.to_epsg() != 4326:
    combined = combined.to_crs(epsg=4326)
for col in combined.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]):
    combined[col] = combined[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
proj = combined.to_crs(epsg=3857)
unioned = proj.geometry.union_all()      
center_pt = unioned.centroid
center_ll = gpd.GeoSeries([center_pt], crs=3857).to_crs(epsg=4326).iloc[0]
m = folium.Map(location=[center_ll.y, center_ll.x], zoom_start=INITIAL_ZOOM)
folium.GeoJson(
    data=combined.__geo_interface__,
    name="all_sequences",
    tooltip=folium.GeoJsonTooltip(fields=[c for c in combined.columns if c != combined.geometry.name])
).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.save(OUTPUT_HTML)
print(f"Map saved to {OUTPUT_HTML}")