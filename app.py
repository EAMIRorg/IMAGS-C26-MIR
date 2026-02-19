from __future__ import annotations

import json
import io
import numpy as np
import pandas as pd
import streamlit as st

from audio_analysis.analyzer import analyze_audio_bytes, AnalyzeOptions



st.set_page_config(page_title="IMAGS-lite Audio Analyzer", layout="wide")
st.title("IMAGS-lite: Audio Feature Analyzer")
st.caption("Tempo (global) + Loudness and Energy trends (time series) for later overlay with GSR.")

uploaded = st.file_uploader("Upload audio (MP3/WAV/etc.)", type=["mp3", "wav", "flac", "m4a", "ogg"])
if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    sr = st.selectbox("Convert sample rate", [44100, 22050], index=0)
with c2:
    smooth_points = st.selectbox("Smoothing points (LUFS)", [1, 3, 5, 9, 15], index=2)


analyze_btn = st.button("Analyze", type="primary")

if not analyze_btn:
    st.info("Click **Analyze** to compute features.")
    st.stop()

opts = AnalyzeOptions(sample_rate=sr, smooth_points=int(smooth_points))

with st.spinner("Running Essentia analysis…"):
    res = analyze_audio_bytes(uploaded.getvalue(), uploaded.name, opts=opts)

meta = res["meta"]
glob = res["global"]
series = res["series"]

# Convert series to dataframe for plotting/export
df = pd.DataFrame({
    "t_sec": series["t_sec"],
    "momentary_lufs": series["momentary_lufs"],
    "energy": series["energy"],
})

# --- Summary metrics
a, b, c, d = st.columns(4)
tempo = glob.get("tempo_bpm", None)
a.metric("Tempo (BPM)", f"{tempo:.1f}" if tempo is not None else "—")
b.metric("Integrated LUFS", f"{glob.get('integrated_lufs'):.1f}" if glob.get("integrated_lufs") is not None else "—")
c.metric("Loudness range (LU)", f"{glob.get('loudness_range_lu'):.1f}" if glob.get("loudness_range_lu") is not None else "—")
d.metric("Duration", f"{meta['duration_sec']:.1f}s")

st.markdown("---")

# --- Plots (all share same x axis)
st.subheader("Loudness over time (Momentary LUFS)")
st.line_chart(df, x="t_sec", y="momentary_lufs")

st.subheader("Energy over time")
st.line_chart(df, x="t_sec", y="energy")

# --- Export
st.markdown("---")
st.subheader("Export")

colx, coly = st.columns(2)

with colx:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download analysis CSV",
        data=csv_bytes,
        file_name=f"{meta['file_name']}.analysis.csv",
        mime="text/csv",
    )

with coly:
    json_bytes = json.dumps(res, indent=2).encode("utf-8")
    st.download_button(
        label="Download analysis JSON",
        data=json_bytes,
        file_name=f"{meta['file_name']}.analysis.json",
        mime="application/json",
    )

