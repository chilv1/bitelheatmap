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

SAVE_DIR = "kmz_exported"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================
# 🔥 GITHUB API UPLOAD FUNCTION
# ===========================================================
def push_to_github(local_file_path, github_path):
    url = f"https://api.github.com/repos/{st.secrets['github']['user']}/{st.secrets['github']['repo']}/contents/{github_path}"

    with open(local_file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    headers = {
        "Authorization": f"token {st.secrets['github']['token']}",
        "Content-Type": "application/json",
    }

    data = {
        "message": f"Add KMZ {github_path}",
        "content": content,
        "branch": st.secrets["github"]["branch"]
    }

    res = requests.put(url, headers=headers, data=json.dumps(data))

    if res.status_code in (200, 201):
        st.success(f"✔ Pushed to GitHub: {github_path}")
        return True
    else:
        st.error(f"❗ GitHub Error {res.status_code}")
        st.code(res.text)
        return False


# ===========================================================
# 🌍 HEATMAP GENERATION
# ===========================================================
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def make_glow_colormap(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    glow_factor = 0.6
    r2, g2, b2 = r + (1 - r)*glow_factor, g + (1 - g)*glow_factor, b + (1 - b)*glow_factor

    return LinearSegmentedColormap.from_list("glow_cmap",[
        (r, g, b, 0.30),
        (r2, g2, b2, 0.80)
    ])


def compute_bounds(lon, lat):
    x1, x2 = lon.min(), lon.max()
    y1, y2 = lat.min(), lat.max()
    dx, dy = x2 - x1, y2 - y1
    return (x1 - dx*0.02, x2 + dx*0.02, y1 - dy*0.02, y2 + dy*0.02)


def build_heatmap_layer(df_op, color_hex, xmin, xmax, ymin, ymax):
    lon = df_op[LON_COL].to_numpy()
    lat = df_op[LAT_COL].to_numpy()

    xn = (lon - xmin) / (xmax - xmin + 1e-9)
    yn = (lat - ymin) / (ymax - ymin + 1e-9)
    xi = np.clip((xn*(GRID_RES-1)).astype(int),0,GRID_RES-1)
    yi = np.clip((yn*(GRID_RES-1)).astype(int),0,GRID_RES-1)

    grid = np.zeros((GRID_RES,GRID_RES),dtype=float)
    np.add.at(grid,(yi,xi),1.0)

    heat = gaussian_filter(grid, sigma=RADIUS)
    maxh = np.nanpercentile(heat[heat>0],99.5)
    if np.isnan(maxh) or maxh==0:
        maxh=np.max(heat)

    cutoff = maxh*THRESHOLD_RATIO
    heat[heat<cutoff] = np.nan

    cmap = make_glow_colormap(color_hex)

    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(
        heat,
        origin="lower",
        extent=[xmin,xmax,ymin,ymax],
        cmap=cmap,
        vmin=cutoff,
        vmax=maxh,
        interpolation="bilinear"
    )
    ax.set_axis_off()

    png_file = tempfile.mktemp(suffix=".png")
    fig.savefig(png_file,dpi=300,transparent=True,bbox_inches="tight",pad_inches=0)
    plt.close(fig)
    return png_file


def create_kmz(layers,xmin,xmax,ymin,ymax):
    kml = simplekml.Kml()
    kml.document.name="Heatmap"

    for op_name,png in layers.items():
        g=kml.newgroundoverlay(name=op_name)
        g.icon.href=png.split("/")[-1]
        g.latlonbox.north=ymax
        g.latlonbox.south=ymin
        g.latlonbox.east=xmax
        g.latlonbox.west=xmin

    kml_file=tempfile.mktemp(suffix=".kml")
    kml.save(kml_file)

    kmz_file=tempfile.mktemp(suffix=".kmz")
    with zipfile.ZipFile(kmz_file,"w",zipfile.ZIP_DEFLATED) as z:
        z.write(kml_file,"doc.kml")
        for png in layers.values():
            z.write(png,os.path.basename(png))

    return kmz_file


# ===========================================================
# 🏁 STREAMLIT INTERFACE
# ===========================================================

st.title("📡 Bitel Heatmap Generator + GitHub archive")


# ===========================================================
# STEP 1 — Generate KMZ
# ===========================================================
st.header("1️⃣ Generate new KMZ")

uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)

if st.button("Generate & Save KMZ"):

    dfs=[]
    for f in uploaded_files:
        df=pd.read_csv(f,sep=None,engine="python")
        df.columns=df.columns.str.lower().str.strip()
        df[OPERATOR_COL]=df[OPERATOR_COL].astype(str).str.upper().str.strip()
        dfs.append(df[[LAT_COL,LON_COL,OPERATOR_COL]].dropna())

    df_all=pd.concat(dfs,ignore_index=True)

    lon=df_all[LON_COL].to_numpy()
    lat=df_all[LAT_COL].to_numpy()
    xmin,xmax,ymin,ymax=compute_bounds(lon,lat)

    layers={}

    for op in df_all[OPERATOR_COL].unique():
        df_op=df_all[df_all[OPERATOR_COL]==op]
        hex_color=OPERATOR_COLORS.get(op,"#808080")
        png=build_heatmap_layer(df_op,hex_color,xmin,xmax,ymin,ymax)
        layers[op]=png

    kmz=create_kmz(layers,xmin,xmax,ymin,ymax)

    timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path=f"{SAVE_DIR}/heatmap_{timestamp}.kmz"
    shutil.copy(kmz,new_path)

    st.success(f"Saved: {new_path}")

    github_path=new_path
    push_to_github(new_path,github_path)


# ===========================================================
# STEP 2 — KMZ archive
# ===========================================================
st.header("2️⃣ KMZ Archive & Download")

kmz_files=[f"{SAVE_DIR}/{f}" for f in os.listdir(SAVE_DIR) if f.endswith(".kmz")]

if not kmz_files:
    st.info("No KMZ files exist yet.")
else:
    selected=st.selectbox("Select KMZ", kmz_files)

    st.download_button("⬇️ Download",
        open(selected,"rb"),
        file_name=os.path.basename(selected)
    )

    st.write("🔗 Public link:")
    st.code(f"https://raw.githubusercontent.com/{st.secrets['github']['user']}/{st.secrets['github']['repo']}/main/{selected}")


# ===========================================================
# STEP 3 — Preview map
# ===========================================================
st.header("3️⃣ Map Preview")

if st.button("Preview KMZ on map"):

    layers = load_kmz_layers(selected)

    m = folium.Map(location=[-12,-77], zoom_start=11)

    for op,png in layers.items():

        group=folium.FeatureGroup(name=op,overlay=True)
        group.add_to(m)

        with open(png, 'rb') as f:
            img_base64=base64.b64encode(f.read()).decode('utf-8')

        folium.raster_layers.ImageOverlay(
            name=op,
            image="data:image/png;base64,"+img_base64,
            bounds=[[-90,-180],[90,180]],
            opacity=0.65,
        ).add_to(group)

    folium.LayerControl().add_to(m)
    st_folium(m,use_container_width=True)
