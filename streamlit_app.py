import streamlit as st
import pandas as pd
import numpy as np
import simplekml
import zipfile
import tempfile
import os
import datetime
import base64
import requests
import json
import shutil


# ====================
# CONFIG
# ====================
LAT_COL = "gps_latitude"
LON_COL = "gps_longitude"
OPERATOR_COL = "carrier"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "kmz_exported")
os.makedirs(SAVE_DIR, exist_ok=True)


# ====================
# CHECK GITHUB SECRETS FIRST
# ====================
if "github" not in st.secrets:
    st.error("🚨 GitHub API SECRET NOT CONFIGURED")
    st.stop()

GH_TOKEN  = st.secrets["github"]["token"]
GH_USER   = st.secrets["github"]["user"]
GH_REPO   = st.secrets["github"]["repo"]
GH_BRANCH = st.secrets["github"]["branch"]


# ====================
# PUSH TO GITHUB
# ====================
def upload_file_to_github(local_file_path, github_repo_path):
    url = f"https://api.github.com/repos/{GH_USER}/{GH_REPO}/contents/{github_repo_path}"

    with open(local_file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    headers = {"Authorization": f"token {GH_TOKEN}"}

    data = {
        "message": f"Add KMZ {github_repo_path}",
        "content": encoded,
        "branch": GH_BRANCH
    }

    response = requests.put(url, headers=headers, data=json.dumps(data))

    if response.status_code in [200, 201]:
        st.success("✔ KMZ successfully uploaded to GitHub!")
    else:
        st.error(f"❗ GitHub upload ERROR {response.status_code}")
        st.code(response.text)


# ====================
# UI
# ====================
st.title("KMZ GENERATOR TEST – only upload to GitHub")

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if st.button("Generate KMZ + Save to GitHub"):

    if not uploaded_files:
        st.error("❗ Please upload CSV files")
        st.stop()

    dfs = []
    for f in uploaded_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower().str.strip()
        dfs.append(df[[LAT_COL, LON_COL, OPERATOR_COL]].dropna())

    df_all = pd.concat(dfs, ignore_index=True)

    # Create KML
    kml = simplekml.Kml()
    kml.newpoint(name="Test", coords=[(0,0)])

    kml_file = tempfile.mktemp(suffix=".kml")
    kml.save(kml_file)

    kmz_file = tempfile.mktemp(suffix=".kmz")
    with zipfile.ZipFile(kmz_file, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(kml_file, "doc.kml")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_kmz = os.path.join(SAVE_DIR, f"heatmap_{timestamp}.kmz")
    shutil.copy(kmz_file, final_kmz)

    st.success(f"Local KMZ saved: {final_kmz}")

    github_path = f"kmz_exported/heatmap_{timestamp}.kmz"
    upload_file_to_github(final_kmz, github_path)
