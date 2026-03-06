import time
import threading

from collections import deque
from random import Random
from typing import final

from math import floor, ceil
from .gsr_reader import GSRConfig

import numpy as np

VIRTUAL_INTERVAL = 50 / 1000 # seconds

def get_seconds():
    return time.time()

@final
class Perlin:
    """ Implements a good-enough approximation of perlin noise for testing.  """

    def __init__(self, scale: float = 8.0, sampleCount: int = 64):
        self.rng = Random()
        self.scale = scale
        self.sampleCount = sampleCount
        self.buffer = list([self.rng.random() for _ in range(sampleCount)])
    
    def sample(self, x: float) -> float:
        x *= self.scale
        i1 = floor(x) % self.sampleCount
        i2 = ceil(x) % self.sampleCount
        if i1 == i2: return self.buffer[i1]
        return self._smootherstep(self.buffer[i1], self.buffer[i2], x - floor(x))

    def _smootherstep(self, a: float, b: float, x: float) -> float:
        if x <= 0.0: return a
        if x >= 1.0: return b
        x = x * x * x * (x * (6 * x - 15) + 10)
        return b * x + a * (1 - x)

@final
class GSRVirtualStream:
    """
    Background serial reader.
    Stores smoothed samples as (pc_ms, gsr_smooth).
    """

    def __init__(self, cfg: GSRConfig | None = None):
        self.cfg = cfg or GSRConfig()

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._smooth_buf = deque[float](maxlen=self.cfg.smooth_window)
        self._points = deque[tuple[int, float]](maxlen=self.cfg.max_points)
        self._lock = threading.Lock()

        self._perlin_lg = Perlin(0.1, 32)
        self._perlin_sm = Perlin(8.0)
        self._time_start = 0.0

        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._time_start = get_seconds()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def last_error(self) -> str | None:
        return self._last_error

    def get_points(self) -> list[tuple[int, float]]:
        """Snapshot of all currently buffered points (pc_ms, gsr_smooth)."""
        with self._lock:
            return list(self._points)

    def clear(self) -> None:
        with self._lock:
            self._points.clear()
            self._smooth_buf.clear()

    def _run(self) -> None:
        try:
            time.sleep(self.cfg.arduino_reset_delay_s)

            while not self._stop.is_set():
                time.sleep(VIRTUAL_INTERVAL)
                
                seconds = get_seconds() - self._time_start
                gsr = 1.5 + self._perlin_lg.sample(seconds) * 2.0 + (self._perlin_sm.sample(seconds) * 1.0 - 0.5);

                self._smooth_buf.append(gsr)
                smooth = float(np.mean(self._smooth_buf))
                now_ms = int(time.time() * 1000)
                # print(gsr, '\t', smooth)

                with self._lock:
                    self._points.append((now_ms, smooth))

        except Exception as e:
            self._last_error = repr(e)
