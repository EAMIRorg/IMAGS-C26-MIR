from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

import numpy as np
import serial
from serial.tools import list_ports


@dataclass
class GSRConfig:
    baud: int = 9600
    timeout_s: float = 1.0
    arduino_reset_delay_s: float = 2.0

    # Smoothing
    smooth_window: int = 25  # samples in moving average

    # Buffering
    max_points: int = 60 * 75  


def list_serial_ports() -> List[Tuple[str, str]]:
    """Return [(device, description), ...]."""
    ports = list(list_ports.comports())
    return [(p.device, p.description) for p in ports]


def parse_line_to_float(line: str) -> Optional[float]:
    s = line.strip()
    if not s:
        return None
    if s.lower() in ("start", "stop"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


class GSRStream:
    """
    Background serial reader.
    Stores smoothed samples as (pc_ms, gsr_smooth).
    """

    def __init__(self, port: str, cfg: Optional[GSRConfig] = None):
        self.port = port
        self.cfg = cfg or GSRConfig()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser: Optional[serial.Serial] = None

        self._smooth_buf = deque(maxlen=self.cfg.smooth_window)
        self._points = deque(maxlen=self.cfg.max_points)
        self._lock = threading.Lock()

        self._last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def last_error(self) -> Optional[str]:
        return self._last_error

    def get_points(self) -> List[Tuple[int, float]]:
        """Snapshot of all currently buffered points (pc_ms, gsr_smooth)."""
        with self._lock:
            return list(self._points)

    def clear(self) -> None:
        with self._lock:
            self._points.clear()
            self._smooth_buf.clear()

    def _run(self) -> None:
        try:
            self._ser = serial.Serial(self.port, self.cfg.baud, timeout=self.cfg.timeout_s)
            time.sleep(self.cfg.arduino_reset_delay_s)
            try:
                self._ser.reset_input_buffer()
            except Exception:
                pass

            while not self._stop.is_set():
                raw = self._ser.readline().decode(errors="ignore").strip()
                gsr = parse_line_to_float(raw)
                if gsr is None:
                    continue

                self._smooth_buf.append(gsr)
                smooth = float(np.mean(self._smooth_buf))
                #print(smooth)
                now_ms = int(time.time() * 1000)

                with self._lock:
                    self._points.append((now_ms, smooth))

        except Exception as e:
            self._last_error = repr(e)
        finally:
            if self._ser:
                try:
                    self._ser.close()
                except Exception:
                    pass