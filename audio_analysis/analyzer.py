from __future__ import annotations
from dataclasses import dataclass 
from typing import Dict, Any, Optional, Tuple

import os
import subprocess
import tempfile
import numpy as np

import essentia 
import essentia.standard as es
#################################################
# Helpers/Utilities
#################################################
def _run_cmd(cmd: list) -> Tuple[int, str, str]:
    """It does what it says on the box"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def _to_wav_stereo(input_path: str, sr: int = 44100) -> str:
    """Convert any input audio into stereo WAV at sample rate 44100. Returns path to temp WAV file"""

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "2", "-ar", str(sr), "-vn", wav_path]
    rc, _out, err = _run_cmd(cmd)

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err}")
    return wav_path

def _moving_average(x: np.ndarray, points: int) -> np.ndarray:
    if points <= 1:
        return x 
    kernel = np.ones(points, dtype=float) / float(points)
    return np.convolve(x, kernel, mode="same")

def _normalize_0_1(x: np.ndarray) -> np.ndarray:
    """To normalize LUFS range to 0-1 for energy"""
    x = x.astype(float)
    mask = np.isfinite(x)
    if not mask.any():
        return np.full_like(x, np.nan)
    xmin = np.nanmin(x[mask])
    xmax = np.nanmax(x[mask])
    if np.isclose(xmax, xmin):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def _safe_float(x) -> Optional[float]:
    """safety"""
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None
    


@dataclass
class AnalyzeOptions:
    sample_rate: int = 44100            # Standard sr
    smooth_points: int = 3              # 3 is good cause gives intricate rhythm indications without being too fuzzy (1 is too noisy)
   

def analyze_audio_bytes(
    file_bytes: bytes,
    file_name: str,
    opts: Optional[AnalyzeOptions] = None,
) -> Dict[str, Any]:
    """
    Analyze an audio file (mp3/wav/etc as bytes) using Essentia.
    USE A WAV FILE FOR INPUT, OTHER FORMATS GET COMPRESSED AND QUIET 

    Loudness: momentary LUFS time-series (optionally smoothed)
    Energy:  smooth RMS-based energy curve (0–1), aligned to loudness time grid
    Tempo: global estimate (if available)
    """
    opts = opts or AnalyzeOptions()

    # ---- Energy settings ----
    energy_window_s = 1.5   # bigger window => smoother "overall energy"
    energy_hop_s = 0.25     # update 4 times per second (still smooth)
    energy_smooth_s = 2.0   # extra smoothing on top of RMS (seconds)
    energy_floor_db = -80.0 # clamp extreme silence before normalization

    # Write upload to temp input
    suffix = os.path.splitext(file_name)[1].lower() or ".bin"
    fd_in, tmp_in = tempfile.mkstemp(suffix=suffix)
    os.close(fd_in)
    with open(tmp_in, "wb") as f:
        f.write(file_bytes)

    tmp_wav = None
    try:
        # Convert to stereo WAV at target sample rate
        tmp_wav = _to_wav_stereo(tmp_in, sr=opts.sample_rate)

        # Load stereo audio
        audio_stereo, fs, nchan, *_ = es.AudioLoader(filename=tmp_wav)()
        fs = int(fs)
        if audio_stereo is None or len(audio_stereo) < fs:
            raise RuntimeError("Audio failed to load or is too short.")
        duration_s = float(len(audio_stereo) / fs)

        # Mono downmix for tempo + energy (average channels)
        if isinstance(audio_stereo, np.ndarray) and audio_stereo.ndim == 2 and audio_stereo.shape[1] >= 2:
            audio_mono = audio_stereo.mean(axis=1).astype(np.float32)
        else:
            audio_mono = np.asarray(audio_stereo, dtype=np.float32)

        # ----------------------------
        # Loudness (Momentary LUFS)
        # ----------------------------
        loud = es.LoudnessEBUR128()
        momentary, _short_term, integrated, loudness_range = loud(audio_stereo)

        momentary_lufs = np.asarray(momentary, dtype=float)
        momentary_lufs = np.clip(momentary_lufs, -70, None)
        if momentary_lufs.ndim == 0 or momentary_lufs.size < 2:
            raise RuntimeError("Momentary loudness series was not produced.")

        momentary_lufs_sm = _moving_average(momentary_lufs, opts.smooth_points)

        # Time axis for loudness series (uniform spacing across duration)
        t_sec = np.linspace(0.0, duration_s, num=momentary_lufs_sm.size, endpoint=False)

        # ----------------------------
        # Energy (smooth RMS-based curve)
        #
        # Compute RMS over multi-second windows so spikes in loudness
        # don't dominate. Then smooth further and normalize to 0–1. (looks prettier for demo and easier to visually analyze)
        #
        # Finally, interpolate onto the loudness t_sec grid so plots align.
        # ----------------------------
        win = max(1, int(round(energy_window_s * fs)))
        hop = max(1, int(round(energy_hop_s * fs)))

        rms_db = []
        t_energy = []

        n = int(audio_mono.size)
        if n < win:
            # fallback: treat whole clip as one window
            frame = audio_mono
            rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
            val_db = 20.0 * np.log10(rms + 1e-12)
            rms_db = np.array([val_db], dtype=float)
            t_energy = np.array([duration_s * 0.5], dtype=float)
        else:
            for start in range(0, n - win + 1, hop):
                frame = audio_mono[start:start + win]
                # RMS (power proxy)
                rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                val_db = 20.0 * np.log10(rms + 1e-12)
                rms_db.append(val_db)
                # time at window center
                t_energy.append((start + 0.5 * win) / fs)

            rms_db = np.asarray(rms_db, dtype=float)
            t_energy = np.asarray(t_energy, dtype=float)

        # Clamp extreme silence so normalization isn't dominated by -inf-ish values
        rms_db = np.maximum(rms_db, energy_floor_db)

        # Extra smoothing in "seconds" (convert to points on energy grid)
        if t_energy.size >= 2:
            dt_e = float(t_energy[1] - t_energy[0])
        else:
            dt_e = energy_hop_s
        smooth_pts_e = max(1, int(round(energy_smooth_s / max(dt_e, 1e-6))))
        rms_db_sm = _moving_average(rms_db, smooth_pts_e)

        # Normalize to 0–1 where 0 ~ quiet, 1 ~ loud/energetic
        energy_curve = _normalize_0_1(rms_db_sm)

        # Interpolate energy onto the loudness time grid (same length as LUFS)
        if t_energy.size >= 2:
            energy_on_lufs_grid = np.interp(t_sec, t_energy, energy_curve, left=0.0, right=0.0)
        else:
            # degenerate fallback
            energy_on_lufs_grid = np.full_like(t_sec, float(energy_curve[0]) if energy_curve.size else 0.0)


        # ----------------------------
        # Tempo estimate (global)
        # ----------------------------
        bpm = None
        bpm_method = "Unavailable"
        try:
            rhythm = es.RhythmExtractor2013(method="multifeature")
            bpm_val, *_ = rhythm(audio_mono)
            bpm = _safe_float(bpm_val)
            bpm_method = "RhythmExtractor2013"
        except Exception:
            bpm = None
            bpm_method = "Unavailable"

        return {                               
            "meta": {
                "file_name": file_name,
                "sample_rate": fs,
                "channels": int(nchan),
                "duration_sec": duration_s,
                "smooth_points": int(opts.smooth_points),
                "energy_window_s": float(energy_window_s),
                "energy_hop_s": float(energy_hop_s),
                "energy_smooth_s": float(energy_smooth_s),
            },
            "global": {
                "tempo_bpm": bpm,                                   
                "tempo_method": bpm_method,
                "integrated_lufs": _safe_float(integrated),
                "loudness_range_lu": _safe_float(loudness_range),
            },
            "series": {
                "t_sec": t_sec.astype(float).tolist(),
                "momentary_lufs": momentary_lufs_sm.astype(float).tolist(),
                "energy": energy_on_lufs_grid.astype(float).tolist(),
            },
        }

    finally:
        for p in [tmp_in, tmp_wav]:
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
