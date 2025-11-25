import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import tempfile
import zipfile
import simplekml

# -------------------- CONFIG --------------------
GRID_RES = 2000
RADIUS = 30
THRESHOLD_RATIO = 0.3
LAT_COL = "gps_latitude"
LON_COL = "gps_longitude"
OPERATOR_COL = "carrier"

OPERATOR_COLORS = {
    "ENTEL":    "#0057A4",
    "MOVISTAR": "#00A65A",
    "CLARO":    "#D40000",
    "BITEL":    "#FFD500"
}

# -------------------- COLORMAP GLOW --------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def make_glow_colormap(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    glow_factor = 0.6
    r2 = r + (1.0 - r) * glow_factor
    g2 = g + (1.0 - g) * glow_factor
    b2 = b + (1.0 - b) * glow_factor
    
    colors = [
        (r, g, b, 0.3),  
        (r2, g2, b2, 0.8)
    ]

    return LinearSegmentedColormap.from_list("glow_cmap", colors)

# -------------------- HEATMAP CORE --------------------

def compute_bounds(lon, lat):
    x1, x2 = lon.min(), lon.max()
    y1, y2 = lat.min(), lat.max()
    dx, dy = x2 - x1, y2 - y1
    return (x1 - dx * 0.02, x2 + dx * 0.02, y1 - dy * 0.02, y2 + dy * 0.02)

def build_heatmap_layer(df_op, color_hex, xmin, xmax, ymin, ymax):
    lon = df_op[LON_COL].to_numpy()
    lat = df_op[LAT_COL].to_numpy()

    xn = (lon - xmin) / (xmax - xmin + 1e-9)
    yn = (lat - ymin) / (ymax - ymin + 1e-9)
    xi = np.clip((xn * (GRID_RES - 1)).astype(int), 0, GRID_RES - 1)
    yi = np.clip((yn * (GRID_RES - 1)).astype(int), 0, GRID_RES - 1)

    grid = np.zeros((GRID_RES, GRID_RES), dtype=float)
    np.add.at(grid, (yi, xi), 1.0)

    heat = gaussian_filter(grid, sigma=RADIUS)
    maxh = np.nanpercentile(heat[heat > 0], 99.5)

    if np.isnan(maxh) or maxh == 0:
        maxh = np.max(heat)

    cutoff = maxh * THRESHOLD_RATIO
    heat[heat < cutoff] = np.nan

    cmap = make_glow_colormap(color_hex)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        heat,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        interpolation="bilinear",
        vmin=cutoff,
        vmax=maxh
    )
    ax.set_axis_off()

    png_file = tempfile.mktemp(suffix=".png")
    fig.savefig(png_file, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return png_file

def create_kmz(layers, xmin, xmax, ymin, ymax):
    kml = simplekml.Kml()
    kml.document.name = "Operators Density Heatmaps"

    for op_name, png in layers.items():
        g = kml.newgroundoverlay(name=op_name)
        g.icon.href = png.split("/")[-1]  
        g.latlonbox.north = ymax
        g.latlonbox.south = ymin
        g.latlonbox.east = xmax
        g.latlonbox.west = xmin

    kml_file = tempfile.mktemp(suffix=".kml")
    kml.save(kml_file)

    kmz_file = tempfile.mktemp(suffix=".kmz")
    with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(kml_file, "doc.kml")
        for png in layers.values():
            z.write(png, png.split("/")[-1])

    return kmz_file

# ============================= STREAMLIT UI =============================

st.title("📡 Geo Heatmap KMZ Generator")

uploaded_files = st.file_uploader(
    "Upload one or multiple CSV files",
    accept_multiple_files=True,
    type=["csv"]
)

if uploaded_files and st.button("Generate KMZ"):
    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f, sep=None, engine="python")
        df.columns = df.columns.str.lower().str.strip()
        df[OPERATOR_COL] = df[OPERATOR_COL].astype(str).str.upper().str.strip()
        dfs.append(df[[LAT_COL, LON_COL, OPERATOR_COL]].dropna())

    df_all = pd.concat(dfs, ignore_index=True)

    lon = df_all[LON_COL].to_numpy()
    lat = df_all[LAT_COL].to_numpy()

    xmin, xmax, ymin, ymax = compute_bounds(lon, lat)

    layers = {}

    for op in df_all[OPERATOR_COL].unique():
        df_op = df_all[df_all[OPERATOR_COL] == op]
        hex_color = OPERATOR_COLORS.get(op, "#808080")
        png = build_heatmap_layer(df_op, hex_color, xmin, xmax, ymin, ymax)
        layers[op] = png

    kmz = create_kmz(layers, xmin, xmax, ymin, ymax)

    st.success("KMZ generated successfully!")

    with open(kmz, "rb") as f:
        st.download_button(
            label="⬇️ Download KMZ",
            data=f,
            file_name="operators_heatmap_glow.kmz",
            mime="application/vnd.google-earth.kmz"
        )
