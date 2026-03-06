from __future__ import annotations

"""
IMAGS-lite Streamlit App

"""

from logging import basicConfig, DEBUG
basicConfig(level=DEBUG)

import time
import json

import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh

from audio_analysis.analyzer import analyze_audio_bytes, AnalyzeOptions
from gsr.gsr_reader import VIRTUAL_GSR, GSRStream, GSRConfig, list_serial_ports
from gsr.gsr_faker import GSRVirtualStream

# ------------------------------------------------------------
# Streamlit config (must be the first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(page_title="IMAGS-lite (Audio + GSR)", layout="wide")


# ------------------------------------------------------------
# Session state init
# ------------------------------------------------------------
# Audio analysis persistence
if "audio_res" not in st.session_state:
    st.session_state.audio_res = None
if "audio_df" not in st.session_state:
    st.session_state.audio_df = None

# Serial port selection persistence
if "gsr_port" not in st.session_state:
    st.session_state.gsr_port = None

# GSR stream object (we only run it during playback)
if "gsr_stream" not in st.session_state:
    st.session_state.gsr_stream = None

# Playback session state
if "playback_started" not in st.session_state:
    st.session_state.playback_started = False
if "playback_start_ms" not in st.session_state:
    st.session_state.playback_start_ms = None

# Frozen audio results for the playback session (so reruns don't change what we're comparing against)
if "playback_audio_res" not in st.session_state:
    st.session_state.playback_audio_res = None
if "playback_audio_df" not in st.session_state:
    st.session_state.playback_audio_df = None

# Captured full-resolution GSR data for the playback session (for correct final plot / export)
if "playback_gsr_df" not in st.session_state:
    st.session_state.playback_gsr_df = None

# Performance: track whether any new samples arrived since last refresh
if "playback_last_pts_len" not in st.session_state:
    st.session_state.playback_last_pts_len = 0

# Performance: keep a downsampled copy used ONLY for plotting
if "playback_gsr_plot_df" not in st.session_state:
    st.session_state.playback_gsr_plot_df = pd.DataFrame({"t_sec": [], "gsr": []})

# Optional: keep a sticky UI choice for loudness visibility
if "show_loudness" not in st.session_state:
    st.session_state.show_loudness = False


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def downsample_df(df: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame:
    """
    Downsample a dataframe to at most max_points rows by taking every k-th sample.
    Keeps plotting fast even if we capture many points.

    This does NOT change the stored full-resolution data; it only affects plotting.
    """
    n = len(df)
    if n <= max_points or n == 0:
        return df
    step = max(1, n // max_points)
    return df.iloc[::step].reset_index(drop=True)


def make_energy_gsr_dual_axis_chart(energy_df: pd.DataFrame, gsr_df: pd.DataFrame) -> alt.LayerChart:
    """
    Build a layered Altair chart with:
      - shared x-axis: t_sec (seconds)
      - left y-axis: energy (0..1)
      - right y-axis: gsr voltage (V)

    energy_df columns: ["t_sec", "energy"]
    gsr_df columns:    ["t_sec", "gsr"]
    """
    base = alt.Chart().encode(x=alt.X("t_sec:Q", title="Time (s)"))

    energy_line = base.mark_line(color="blue", strokeWidth=2).encode(
        y=alt.Y("energy:Q", title="Energy (0–1)"),
        tooltip=[
            alt.Tooltip("t_sec:Q", title="t (s)", format=".2f"),
            alt.Tooltip("energy:Q", title="Energy", format=".3f"),
        ],
    ).properties(data=energy_df)

    gsr_line = base.mark_line(color="red", strokeWidth=2).encode(
        y=alt.Y("gsr:Q", title="GSR (V)", axis=alt.Axis(orient="right")),
        tooltip=[
            alt.Tooltip("t_sec:Q", title="t (s)", format=".2f"),
            alt.Tooltip("gsr:Q", title="GSR (V)", format=".3f"),
        ],
    ).properties(data=gsr_df)

    return (
        alt.layer(energy_line, gsr_line)
        .resolve_scale(y="independent")
        .properties(height=360)
    )


def stop_and_clear_stream() -> None:
    """Stop any active GSR stream and clear it from session state."""
    if st.session_state.gsr_stream is not None:
        try:
            st.session_state.gsr_stream.stop()
        except Exception:
            pass
    st.session_state.gsr_stream = None


def reset_playback_state() -> None:
    """Reset playback-related state and stop stream."""
    st.session_state.playback_started = False
    st.session_state.playback_start_ms = None
    st.session_state.playback_audio_res = None
    st.session_state.playback_audio_df = None
    st.session_state.playback_gsr_df = None
    st.session_state.playback_last_pts_len = 0
    st.session_state.playback_gsr_plot_df = pd.DataFrame({"t_sec": [], "gsr": []})
    stop_and_clear_stream()


# ------------------------------------------------------------
# Sidebar: Serial port selection ONLY (no manual Start/Stop GSR)
# ------------------------------------------------------------
st.sidebar.header("GSR Setup")

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
    st.sidebar.caption("GSR will start automatically when you click **Start Playback**.")
else:
    st.session_state.gsr_port = None
    st.sidebar.warning("No serial ports found. Plug in Arduino and refresh.")


# ------------------------------------------------------------
# Autorefresh strategy
# ------------------------------------------------------------
# Only refresh during playback so UI stays snappy while uploading/analyzing.
# (The chart updates live while playback is running.)
if st.session_state.playback_started:
    # 400ms is a good compromise: smooth enough visually, lighter CPU/bandwidth than 200ms.
    st_autorefresh(interval=400, key="playback_autorefresh")


# ------------------------------------------------------------
# Main UI header
# ------------------------------------------------------------
st.title("IMAGS-lite: Audio + GSR Overlay Viewer")
st.caption(
    "Demo flow: Analyze song → Start Playback (start song on phone at the same time) → "
    "Energy + GSR overlay runs for song duration → GSR stops automatically → final plot remains."
)


# ------------------------------------------------------------
# Audio Analysis controls
# ------------------------------------------------------------
st.subheader("1) Audio Analysis")

uploaded = st.file_uploader(
    "Upload audio (MP3/WAV/etc.)",
    type=["mp3", "wav", "flac", "m4a", "ogg"],
)

c1, c2, c3 = st.columns(3)
with c1:
    sr = st.selectbox("Convert sample rate", [44100, 22050], index=0)
with c2:
    smooth_points = st.selectbox("Smoothing points (LUFS)", [1, 3, 5, 9, 15], index=2)
with c3:
    clear_all = st.button("Clear / Reset All")

if clear_all:
    st.session_state.audio_res = None
    st.session_state.audio_df = None
    reset_playback_state()

analyze_btn = st.button("Analyze song", type="primary", disabled=(uploaded is None))

if analyze_btn and uploaded is not None:
    opts = AnalyzeOptions(sample_rate=sr, smooth_points=int(smooth_points))
    with st.spinner("Running Essentia analysis…"):
        res = analyze_audio_bytes(uploaded.getvalue(), uploaded.name, opts=opts)

    series = res["series"]

    df = pd.DataFrame(
        {
            "t_sec": series["t_sec"],
            "energy": series["energy"],
            "momentary_lufs": series["momentary_lufs"],
        }
    )

    st.session_state.audio_res = res
    st.session_state.audio_df = df

    # New analysis invalidates any previous playback session
    reset_playback_state()


# ------------------------------------------------------------
# Show analysis metrics (only if audio is analyzed)
# ------------------------------------------------------------
if st.session_state.audio_res is not None:
    res = st.session_state.audio_res
    meta = res["meta"]
    glob = res["globals"]

    a, b, c, d = st.columns(4)
    tempo = glob.get("tempo_bpm")
    a.metric("Tempo (BPM)", f"{tempo:.1f}" if tempo is not None else "—")
    b.metric(
        "Integrated LUFS",
        f"{glob.get('integrated_lufs'):.1f}" if glob.get("integrated_lufs") is not None else "—",
    )
    c.metric(
        "Loudness range (LU)",
        f"{glob.get('loudness_range_lu'):.1f}" if glob.get("loudness_range_lu") is not None else "—",
    )
    d.metric("Duration", f"{meta['duration_sec']:.1f}s")

st.markdown("---")


# ------------------------------------------------------------
# Playback Sync controls (this is where GSR starts)
# ------------------------------------------------------------
st.subheader("2) Playback Sync (Start song on phone + click Start Playback)")

can_start_playback = (
    st.session_state.audio_res is not None
    and st.session_state.audio_df is not None
    and st.session_state.gsr_port is not None
    and not st.session_state.playback_started
)

colp1, colp2, colp3 = st.columns(3)

with colp1:
    start_playback = st.button("Start Playback", type="primary", disabled=not can_start_playback)

with colp2:
    stop_playback = st.button("Stop Playback Early", disabled=not st.session_state.playback_started)

with colp3:
    reset_playback = st.button("Reset Playback Session")

if reset_playback:
    reset_playback_state()

if start_playback:
    # Freeze audio results for this playback session
    st.session_state.playback_audio_res = st.session_state.audio_res
    st.session_state.playback_audio_df = st.session_state.audio_df.copy()

    # Define t=0 at this click
    st.session_state.playback_start_ms = int(time.time() * 1000)
    st.session_state.playback_started = True

    # Reset performance tracking + captured dataframes
    st.session_state.playback_last_pts_len = 0
    st.session_state.playback_gsr_df = pd.DataFrame({"t_sec": [], "gsr": []})
    st.session_state.playback_gsr_plot_df = pd.DataFrame({"t_sec": [], "gsr": []})

    # Stop any old stream, then start a new one
    stop_and_clear_stream()

    # IMPORTANT: Set max_points large enough to hold the whole song.
    # If you already updated GSRConfig elsewhere, you can remove this.
    duration = float(st.session_state.playback_audio_res["meta"]["duration_sec"])
    # Conservative guess: 80 samples/sec + 10s padding (covers typical Arduino rates safely).
    # Adjust to 120 if you suspect very high sample rate.
    cfg = GSRConfig(max_points=int((duration + 10.0) * 80))

    stream: GSRStream | GSRVirtualStream

    if st.session_state.gsr_port == VIRTUAL_GSR:
        stream = GSRVirtualStream(cfg)
    else:
        stream = GSRStream(st.session_state.gsr_port, cfg)

    stream.start()

    # Clear any pre-start buffered points so overlay begins clean at t=0
    try:
        stream.clear()
    except Exception:
        pass

    st.session_state.gsr_stream = stream

if stop_playback:
    # Stop early but keep captured plot/data on screen
    st.session_state.playback_started = False
    stop_and_clear_stream()

st.markdown("---")


# ------------------------------------------------------------
# 3) The ONE main plot: Energy + GSR overlay (dual-axis)
# ------------------------------------------------------------
st.subheader("3) Energy + GSR Overlay (Dual-axis)")

if st.session_state.playback_audio_res is None or st.session_state.playback_audio_df is None:
    st.info("Analyze a song, then click **Start Playback** to begin the synced Energy + GSR overlay.")
else:
    duration = float(st.session_state.playback_audio_res["meta"]["duration_sec"])
    energy_df_full = st.session_state.playback_audio_df[["t_sec", "energy"]].copy()

    # If playback is active, update captured GSR only if new samples arrived
    if st.session_state.playback_started:
        if st.session_state.playback_start_ms is None:
            st.warning("Playback started but start time is missing. Reset playback and try again.")
        elif st.session_state.gsr_stream is None:
            st.warning("Playback started but GSR stream is not running. Reset playback and try again.")
        else:
            pts = st.session_state.gsr_stream.get_points()  # [(pc_ms, voltage), ...]
            pts_len = len(pts)

            # Rebuild only if new points arrived since last refresh
            if pts_len > st.session_state.playback_last_pts_len and pts_len > 0:
                st.session_state.playback_last_pts_len = pts_len
                t0_ms = int(st.session_state.playback_start_ms)

                # Build full-resolution GSR df
                gsr_df_full = pd.DataFrame(
                    {
                        "t_sec": [(ms - t0_ms) / 1000.0 for ms, _v in pts],
                        "gsr": [v for _ms, v in pts],
                    }
                )

                # Clamp to song window [0, duration]
                gsr_df_full = gsr_df_full[
                    (gsr_df_full["t_sec"] >= 0.0) & (gsr_df_full["t_sec"] <= duration)
                ]

                # Store full-resolution for correctness/persistence
                st.session_state.playback_gsr_df = gsr_df_full

                # Store downsampled plot version for performance
                st.session_state.playback_gsr_plot_df = downsample_df(gsr_df_full, max_points=2500)

            # Auto-stop when song duration elapses
            elapsed = (int(time.time() * 1000) - int(st.session_state.playback_start_ms)) / 1000.0
            if elapsed >= duration:
                st.session_state.playback_started = False
                stop_and_clear_stream()

    # Render the overlay chart using DOWN-SAMPLED data (fast).
    # Full-res remains stored in st.session_state.playback_gsr_df.
    energy_plot_df = downsample_df(energy_df_full, max_points=2500)

    gsr_plot_df = st.session_state.playback_gsr_plot_df
    if gsr_plot_df is None or gsr_plot_df.empty:
        gsr_plot_df = pd.DataFrame({"t_sec": [], "gsr": []})

    chart = make_energy_gsr_dual_axis_chart(energy_plot_df, gsr_plot_df)
    st.altair_chart(chart, width="stretch")

    # Helpful narration / debugging
    if st.session_state.playback_start_ms is not None:
        elapsed = (int(time.time() * 1000) - int(st.session_state.playback_start_ms)) / 1000.0
        if st.session_state.playback_started:
            st.caption(f"Playback running — elapsed {elapsed:.1f}s / {duration:.1f}s (t=0 at Start Playback click)")
        else:
            st.caption("Playback ended (or stopped) — final overlay captured for the full song window.")

    # Optional: uncomment this if you want to confirm point counts during troubleshooting
    # st.caption(f"GSR captured points (full): {0 if st.session_state.playback_gsr_df is None else len(st.session_state.playback_gsr_df)}")

st.markdown("---")


# ------------------------------------------------------------
# Optional: Loudness plot (toggleable)
# ------------------------------------------------------------
st.subheader("Optional: Loudness (LUFS)")

st.session_state.show_loudness = st.checkbox(
    "Show loudness (Momentary LUFS) plot",
    value=st.session_state.show_loudness,
)

if st.session_state.show_loudness:
    # Prefer frozen playback session data if present
    df_for_lufs = st.session_state.playback_audio_df if st.session_state.playback_audio_df is not None else st.session_state.audio_df
    if df_for_lufs is None:
        st.info("Analyze a song to see loudness.")
    else:
        st.line_chart(df_for_lufs, x="t_sec", y="momentary_lufs")


# ------------------------------------------------------------
# Export (based on the most recent analysis results)
# ------------------------------------------------------------
st.markdown("---")
st.subheader("Export")

if st.session_state.audio_res is None or st.session_state.audio_df is None:
    st.info("Analyze a song to enable export.")
else:
    res = st.session_state.audio_res
    df = st.session_state.audio_df
    meta = res["meta"]

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