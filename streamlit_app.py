import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import tempfile
import zipfile
import simplekml
import base64
import requests
from datetime import datetime

# ================== GITHUB CONFIG ==================
GITHUB_USER = st.secrets["github_user"]
GITHUB_REPO = st.secrets["github_repo"]
GITHUB_BRANCH = st.secrets["github_branch"]
GITHUB_TOKEN = st.secrets["github_token"]

# ================== UPLOADER ==================
def upload_kmz_to_github(local_file_path):
    with open(local_file_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"heatmap_{timestamp}.kmz"
    remote_path = f"kmz_exported/{filename}"

    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{remote_path}"

    data = {
        "message": f"Upload KMZ {filename}",
        "content": encoded,
        "branch": GITHUB_BRANCH
    }
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    r = requests.put(url, json=data, headers=headers)

    if r.status_code in [200, 201]:
        j = r.json()
        return j["content"]["html_url"]

    st.error(f"❗ GitHub upload failed: {r.text}")
    return None


# ================== HEATMAP CONFIG ==================
GRID_RES = 2000
RADIUS = 30
THRESHOLD_RATIO = 0.3

OPERATOR_COLORS = {
    "ENTEL":    "#0057A4",
    "MOVISTAR": "#00A65A",
    "CLARO":    "#D40000",
    "BITEL":    "#FFD500"
}


# ================== COLOR MAP ==================
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def make_glow_colormap(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    r2 = r + (1.0 - r) * 0.6
    g2 = g + (1.0 - g) * 0.6
    b2 = b + (1.0 - b) * 0.6
    return LinearSegmentedColormap.from_list("glow", [(r, g, b, 0.3), (r2, g2, b2, 0.8)])


# ================== HEATMAP CORE ==================
def compute_bounds(lon, lat):
    return (lon.min(), lon.max(), lat.min(), lat.max())


def build_heatmap_layer(df_op, color_hex, xmin, xmax, ymin, ymax):
    lon = df_op["gps_longitude"].to_numpy()
    lat = df_op["gps_latitude"].to_numpy()

    xn = (lon - xmin) / (xmax - xmin + 1e-9)
    yn = (lat - ymin) / (ymax - ymin + 1e-9)

    xi = np.clip((xn * 1999).astype(int), 0, 1999)
    yi = np.clip((yn * 1999).astype(int), 0, 1999)

    grid = np.zeros((2000, 2000), float)
    np.add.at(grid, (yi, xi), 1)

    heat = gaussian_filter(grid, sigma=RADIUS)
    cutoff = np.max(heat) * THRESHOLD_RATIO
    heat[heat < cutoff] = np.nan

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(heat, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap=make_glow_colormap(color_hex))
    ax.set_axis_off()

    png = tempfile.mktemp(".png")
    fig.savefig(png, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return png


def create_kmz(layers, xmin, xmax, ymin, ymax):
    kml = simplekml.Kml()
    for op_name, png in layers.items():
        g = kml.newgroundoverlay(name=op_name)
        g.icon.href = png.split("/")[-1]
        g.latlonbox.north = ymax
        g.latlonbox.south = ymin
        g.latlonbox.east = xmax
        g.latlonbox.west = xmin

    kml_file = tempfile.mktemp(".kml")
    kmz_file = tempfile.mktemp(".kmz")

    kml.save(kml_file)
    with zipfile.ZipFile(kmz_file, "w") as z:
        z.write(kml_file, "doc.kml")
        for png in layers.values():
            z.write(png, png.split("/")[-1])

    return kmz_file


# ================== UI ==================
st.title("📡 Heatmap Generator + GitHub Archive")

files = st.file_uploader("Upload CSV(s)", accept_multiple_files=True)

if files and st.button("Generate"):
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        df.columns = df.columns.str.lower().str.strip()
        dfs.append(df[["gps_latitude","gps_longitude","carrier"]].dropna())

    df_all = pd.concat(dfs)
    df_all["carrier"] = df_all["carrier"].str.replace('"','').str.upper()

    lon = df_all["gps_longitude"].to_numpy()
    lat = df_all["gps_latitude"].to_numpy()

    xmin, xmax, ymin, ymax = compute_bounds(lon, lat)
    layers = {}

    for op in df_all["carrier"].unique():
        png = build_heatmap_layer(df_all[df_all["carrier"] == op], OPERATOR_COLORS.get(op, "#888"), xmin, xmax, ymin, ymax)
        layers[op] = png

    kmz = create_kmz(layers, xmin, xmax, ymin, ymax)

    st.success("KMZ created")

    with open(kmz, "rb") as f:
        st.download_button("⬇️ Download KMZ", f, "heatmap.kmz")

    st.info("Uploading to GitHub…")

    url = upload_kmz_to_github(kmz)

    if url:
        st.success("✔ Uploaded")
        st.markdown(f"🔗 {url}")
    else:
        st.error("❗ Upload failed")
