from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from audio_analysis.analyzer import analyze_audio_bytes, AnalyzeOptions

# NEW: import your GSR module
from gsr.gsr_reader import GSRStream, GSRConfig, list_serial_ports


# ----------------------------
# Session state init (NEW)
# ----------------------------
if "gsr_stream" not in st.session_state:
    st.session_state.gsr_stream = None
if "gsr_port" not in st.session_state:
    st.session_state.gsr_port = None


# ----------------------------
# Sidebar: GSR Controls (NEW)
# ----------------------------
st.sidebar.header("GSR (Live)")

ports = list_serial_ports()  # [(device, description), ...]
port_labels = [f"{dev} — {desc}" for dev, desc in ports]
port_devices = [dev for dev, _desc in ports]

selected_idx = 0
if st.session_state.gsr_port in port_devices:
    selected_idx = port_devices.index(st.session_state.gsr_port)

if ports:
    chosen_label = st.sidebar.selectbox("Serial port", port_labels, index=selected_idx)
    chosen_port = port_devices[port_labels.index(chosen_label)]
    st.session_state.gsr_port = chosen_port
else:
    st.sidebar.warning("No serial ports found. Plug in Arduino and refresh.")
    chosen_port = None

colA, colB = st.sidebar.columns(2)

start_clicked = colA.button("Start GSR", disabled=(chosen_port is None))
stop_clicked = colB.button("Stop GSR", disabled=(st.session_state.gsr_stream is None))

if start_clicked:
    # stop any existing stream
    if st.session_state.gsr_stream is not None:
        try:
            st.session_state.gsr_stream.stop()
        except Exception:
            pass

    cfg = GSRConfig()  # uses defaults: baud=9600, smooth_window=50, etc.
    stream = GSRStream(chosen_port, cfg)
    stream.start()
    st.session_state.gsr_stream = stream

if stop_clicked and st.session_state.gsr_stream is not None:
    st.session_state.gsr_stream.stop()
    st.session_state.gsr_stream = None

# Status
if st.session_state.gsr_stream is None:
    st.sidebar.info("GSR: not running")
else:
    if st.session_state.gsr_stream.is_running():
        st.sidebar.success(f"GSR: running on {st.session_state.gsr_port}")
    else:
        st.sidebar.error("GSR: stopped (thread not running)")
        err = st.session_state.gsr_stream.last_error()
        if err:
            st.sidebar.write("Last error:", err)

# Auto-refresh while GSR is running (NEW)
if st.session_state.gsr_stream is not None and st.session_state.gsr_stream.is_running():
    st.autorefresh(interval=200, key="gsr_autorefresh")  # 5 Hz refresh


# ----------------------------
# Main UI (your existing audio analyzer)
# ----------------------------
st.set_page_config(page_title="IMAGS-lite Audio Analyzer", layout="wide")
st.title("IMAGS-lite: Audio Feature Analyzer")
st.caption("Tempo (global) + Loudness and Energy trends (time series) for later overlay with GSR.")

# NEW: show live GSR plot at top (minimal)
st.subheader("Live GSR (smoothed)")

if st.session_state.gsr_stream is None:
    st.info("Click **Start GSR** in the sidebar to begin streaming.")
else:
    pts = st.session_state.gsr_stream.get_points()  # [(pc_ms, gsr), ...]
    if not pts:
        st.warning("No GSR samples yet…")
    else:
        # Convert to dataframe, time relative to first sample for now
        t0 = pts[0][0]
        gsr_df = pd.DataFrame({
            "t_sec": [(ms - t0) / 1000.0 for ms, _v in pts],
            "gsr": [v for _ms, v in pts],
        })

        # Rolling last 60 seconds
        if len(gsr_df) > 2:
            t_now = gsr_df["t_sec"].iloc[-1]
            gsr_df = gsr_df[gsr_df["t_sec"] >= (t_now - 60.0)]

        st.line_chart(gsr_df, x="t_sec", y="gsr")

st.markdown("---")

# ----- Your existing audio analysis UI -----
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

df = pd.DataFrame({
    "t_sec": series["t_sec"],
    "momentary_lufs": series["momentary_lufs"],
    "energy": series["energy"],
})

a, b, c, d = st.columns(4)
tempo = glob.get("tempo_bpm", None)
a.metric("Tempo (BPM)", f"{tempo:.1f}" if tempo is not None else "—")
b.metric("Integrated LUFS", f"{glob.get('integrated_lufs'):.1f}" if glob.get("integrated_lufs") is not None else "—")
c.metric("Loudness range (LU)", f"{glob.get('loudness_range_lu'):.1f}" if glob.get("loudness_range_lu") is not None else "—")
d.metric("Duration", f"{meta['duration_sec']:.1f}s")

st.markdown("---")

st.subheader("Loudness over time (Momentary LUFS)")
st.line_chart(df, x="t_sec", y="momentary_lufs")

st.subheader("Energy over time")
st.line_chart(df, x="t_sec", y="energy")

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