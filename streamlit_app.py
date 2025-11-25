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
import json
import requests
import shutil



# ==================== CONFIG ====================
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



# ==================== PATH FIX ====================
# ABSOLUTE PATH FIX
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "kmz_exported")
os.makedirs(SAVE_DIR, exist_ok=True)



# ==================== CHECK GITHUB SECRETS ====================
if "github" not in st.secrets:
    st.error("❗ Missing Streamlit Secrets for GitHub API")
    st.stop()


REPO_USER   = st.secrets["github"]["user"]
REPO_NAME   = st.secrets["github"]["repo"]
REPO_BRANCH = st.secrets["github"]["branch"]
GH_TOKEN    = st.secrets["github"]["token"]



# ==================== GITHUB UPLOAD ====================
def push_to_github(local_file_path, github_path):
    url = f"https://api.github.com/repos/{REPO_USER}/{REPO_NAME}/contents/{github_path}"

    with open(local_file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Content-Type": "application/json",
    }

    data = {
        "message": f"Add KMZ {github_path}",
        "content": content,
        "branch": REPO_BRANCH
    }

    res = requests.put(url, headers=headers, data=json.dumps(data))

    if res.status_code in (200, 201):
        st.success(f"✔ Uploaded to GitHub: {github_path}")
    else:
        st.error(f"❗ GitHub upload ERROR {res.status_code}")
        st.code(res.text)



# ==================== DATA PROCESSING ====================
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def make_glow_colormap(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    glow_factor = 0.6
    r2 = r + (1.0 - r) * glow_factor
    g2 = g + (1.0 - g) * glow_factor
    b2 = b + (1.0 - b) * glow_factor

    return LinearSegmentedColormap.from_list("glow_cmap", [
        (r, g, b, 0.3),
        (r2, g2, b2, 0.8)
    ])


def compute_bounds(lon, lat):
    return (lon.min(), lon.max(), lat.min(), lat.max())



# ==================== KMZ LOADER FIX ====================
def load_kmz_layers(kmz_file_path):

    if not os.path.exists(kmz_file_path):
        st.error(f"KMZ does not exist: {kmz_file_path}")
        st.stop()

    tmp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(kmz_file_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    overlays = {}

    for f in os.listdir(tmp_dir):
        if f.endswith(".png"):
            overlays[f.replace(".png", "")] = os.path.join(tmp_dir, f)

    return overlays



# ==================== UI ====================
st.title("📡 Bitel Heatmap Generator & GitHub Archive")



# ==================================================
# STEP 1 — GENERATE KMZ
# ==================================================
st.header("1️⃣ Generate new KMZ")

uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)

if st.button("Generate & Save KMZ"):

    if not uploaded_files:
        st.error("❗ Please upload CSV files")
        st.stop()

    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
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
        png = tempfile.mktemp(suffix=".png")

        plt.figure(figsize=(10, 10))
        plt.scatter(df_op[LON_COL], df_op[LAT_COL], s=0.1)
        plt.axis("off")
        plt.savefig(png, dpi=300, transparent=True)
        plt.close()

        layers[op] = png

    kmz = tempfile.mktemp(suffix=".kmz")

    kml = simplekml.Kml()

    for op, png in layers.items():
        g = kml.newgroundoverlay(name=op)
        g.icon.href = png.split("/")[-1]

    kml_file = tempfile.mktemp(suffix=".kml")
    kml.save(kml_file)

    with zipfile.ZipFile(kmz, "w") as z:
        z.write(kml_file, "doc.kml")
        for png in layers.values():
            z.write(png, os.path.basename(png))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(SAVE_DIR, f"heatmap_{timestamp}.kmz")
    shutil.copy(kmz, final_path)

    st.success(f"✔ KMZ saved: {final_path}")

    push_to_github(final_path, f"kmz_exported/heatmap_{timestamp}.kmz")



# ==================================================
# STEP 2 — FILE LIST
# ==================================================
st.header("2️⃣ KMZ Archive")

kmz_files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".kmz")]

if not kmz_files:
    st.info("❗ No KMZ files yet")
else:
    selected = st.selectbox("Choose KMZ:", kmz_files)



# ==================================================
# STEP 3 — PREVIEW MAP
# ==================================================
st.header("3️⃣ Preview on Map")

if st.button("Preview KMZ on map"):

    if not selected:
        st.error("❗ No KMZ selected")
        st.stop()

    if not os.path.exists(selected):
        st.error(f"❗ File not found: {selected}")
        st.stop()

    st.write(f"📦 Loading KMZ: {selected}")

    layers = load_kmz_layers(selected)

    m = folium.Map(location=[-12, -77], zoom_start=11)

    for op, png in layers.items():

        group = folium.FeatureGroup(name=op)
        group.add_to(m)

        with open(png, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        folium.raster_layers.ImageOverlay(
            image="data:image/png;base64," + img_base64,
            bounds=[[-90, -180], [90, 180]],
            opacity=0.65,
        ).add_to(group)

    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True)
