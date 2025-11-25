import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import tempfile
import zipfile
import simplekml
import folium
from streamlit_folium import st_folium
import base64
import datetime
import os
import shutil
import subprocess


# -------------------- PERMANENT STORAGE --------------------
SAVE_DIR = "kmz_exported"
os.makedirs(SAVE_DIR, exist_ok=True)


# -------------------- GIT PUSH --------------------
def git_save(file):
    subprocess.run(["git", "config", "--global", "user.email", "streamlit.bot@bot"])
    subprocess.run(["git", "config", "--global", "user.name", "Streamlit Bot"])
    subprocess.run(["git", "add", file])
    subprocess.run(["git", "commit", "-m", f"Add KMZ file {file}"])
    subprocess.run(["git", "push"])


def save_kmz_persistently(kmz_path):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = f"{SAVE_DIR}/heatmap_{timestamp}.kmz"
    shutil.copy(kmz_path, new_path)
    git_save(new_path)
    return new_path


def list_saved_kmz_files():
    files = []
    for f in os.listdir(SAVE_DIR):
        if f.endswith(".kmz"):
            files.append(f"{SAVE_DIR}/{f}")
    return sorted(files)


def load_kmz_layers(kmz_file):
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(kmz_file, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    overlays = {}
    for f in os.listdir(tmp_dir):
        if f.endswith(".png"):
            overlays[f.replace(".png", "")] = os.path.join(tmp_dir, f)

    return overlays


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


# -------------------- COLORMAP --------------------
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


# -------------------- APP UI --------------------
st.title("📡 Geo Heatmap KMZ Management & Visualization")


# =============================================================
# 1) Generate KMZ from CSV
# =============================================================
st.header("📍 1) Generate new KMZ From CSV")

uploaded_files = st.file_uploader(
    "Upload CSV",
    accept_multiple_files=True,
    type=["csv"],
)

if st.button("Generate & Save KMZ Permanently"):
    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f, sep=None, engine="python")
        df.columns = df.columns.str.lower().str.strip()
        df[OPERATOR_COL] = df[OPERATOR_COL].astype(str).str.upper().str.strip()
        dfs.append(df[[LAT_COL, LON_COL, OPERATOR_COL]].dropna())

    df_all = pd.concat(dfs, ignore_index=True)

    lon = df_all[LON_COL].to_numpy()
    lat = df_all[LAT_COL].to_numpy()

    xmin, xmax, ymin, ymax = lon.min(), lon.max(), lat.min(), lat.max()

    # build layers
    layers = {}

    for op in df_all[OPERATOR_COL].unique():
        df_op = df_all[df_all[OPERATOR_COL] == op]

        color = OPERATOR_COLORS.get(op, "#999999")
        heat = gaussian_filter(np.histogram2d(
            df_op[ LAT_COL ],
            df_op[ LON_COL ],
            bins=GRID_RES)[0], sigma=RADIUS)

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(heat, cmap="hot", interpolation="bilinear")
        ax.set_axis_off()

        png_path = tempfile.mktemp(suffix=".png")
        fig.savefig(png_path, dpi=300, transparent=True)
        plt.close(fig)

        layers[op] = png_path

    # generate kmz
    kml = simplekml.Kml()
    for op, png in layers.items():
        g = kml.newgroundoverlay(name=op)
        g.icon.href = png.split("/")[-1]

    kml_path = tempfile.mktemp(suffix=".kml")
    kml.save(kml_path)

    kmz_file = tempfile.mktemp(suffix=".kmz")
    with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(kml_path, "doc.kml")
        for png in layers.values():
            z.write(png, os.path.basename(png))

    # SAVE PERMANENT
    final_path = save_kmz_persistently(kmz_file)

    st.success(f"KMZ saved permanently: {final_path}")
    st.success("File also pushed to GitHub")


# =============================================================
# 2) KMZ HISTORY PAGE
# =============================================================
st.header("🕘 2) Heatmap History Archive")

kmz_files = list_saved_kmz_files()

if not kmz_files:
    st.warning("❗No saved KMZ yet")
else:
    selected_kmz = st.selectbox("Select KMZ to preview:", kmz_files)

    public_url = f"https://raw.githubusercontent.com/chilv1/Bitelkmz/main/{selected_kmz}"

    st.write("🔗 Public link:")
    st.code(public_url)

    st.download_button("⬇️ Download KMZ", open(selected_kmz, "rb"), file_name=os.path.basename(selected_kmz))

    if st.button("Preview this KMZ"):
        layers = load_kmz_layers(selected_kmz)

        m = folium.Map(location=[-12, -77], zoom_start=12)

        for op, png in layers.items():
            group = folium.FeatureGroup(name=op, overlay=True)
            group.add_to(m)

            with open(png, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            folium.raster_layers.ImageOverlay(
                image="data:image/png;base64," + img_b64,
                bounds=[[-90,-180],[90,180]],
                opacity=0.65,
            ).add_to(group)

        folium.LayerControl().add_to(m)

        st_folium(m, use_container_width=True)
