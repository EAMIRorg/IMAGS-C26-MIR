from logging import getLogger
logger = getLogger("AudioAnalyzerStream")

from enum import IntEnum
from time import sleep, time
from typing import final, override
from math import floor, ceil

from collections import deque
from dataclasses import dataclass

from pandas import DataFrame
from threading import Thread, Lock

from .player_stream import AudioPlayerStream
from .analyzer import AnalysisResult

# --------------------------------------------------
# Resampling/interpolation utilities
# (!!!!!! THESE ARE CURRENTLY UNUSED !!!!!!)
# --------------------------------------------------

# region Sampling

def bicubic(x: float) -> float:
    """ Defines a bicubic-approximating kernel, supports { -2 < x < 2 } """
    a = -0.5 # catmull-rom
    x = abs(x)
    if x >= 2: return 0
    if x > 1: return a*x*x*x - 5*a*x*x + 8*a*x - 4*a
    return (a + 2) * x*x*x - (a + 3) * x*x + 1

def interpolate_bicubic(arr: list[float], pos: float) -> float:
    """ Gets the value at the given float index in arr via bicubic interpolation, assuming a fixed interval. """
    i1 = max(floor(pos) - 2, 0)
    i2 = min(ceil(pos) + 2, len(arr) - 1)
    weightsum = 0.0
    result = 0.0

    if i2 <= i1:
        return arr[i1]

    for i in range(i1, i2):
        weight = bicubic(i - pos)
        weightsum += weight
        result += weight * arr[i]

    if weightsum < 0.0001: return 0.0
    return float(result / weightsum)

def interpolate_linear(arr: list[float], pos: float) -> float:
    """ Standard linear interpolation, worse than the above for approximation. Also assumes a fixed interval. """
    i1 = floor(pos)
    i2 = ceil(pos)
    if i1 == i2 or i2 >= len(arr): return arr[i1]
    mix = (pos - i1)
    return mix * arr[i2] + (1 - mix) * arr[i1]

# NOTE(jadon): This is just a debug function that I made to test interpolation, ignore.
# def fmt_points(arr: list[float] | list[int], ndigits: int = 4) -> list[tuple[float, float]]:
#     m = len(arr) - 1
#     return [
#         (i / m, round(float(x), ndigits)) for i, x in enumerate(arr)
#     ]

# endregion Sampling

# --------------------------------------------------
# Audio analysis class
# --------------------------------------------------

# region Sync-stream

class UserAction(IntEnum):
    Empty = 0
    Play = 1
    Stop = 2

@dataclass
class AnalysisPlayerOptions:
    onehot_falloff: float = 0.8
    """ Every frame, onehots will equal max(current_frame, previous_frame * X) """
    max_points: int = 30 * 30
    """ The maximum number of points to keep in memory at any given time. """
    push_timeout: float = 0.0333
    """ The minimum time, in seconds, between _push_frame being called. This rate may not always be met! """

@final
class AnalysisPlayerSeries:
    def __init__(self, maxlen: int) -> None:
        self.time_seconds = deque[float](maxlen=maxlen)
        self.onehot_user_play = deque[float](maxlen=maxlen)
        self.onehot_user_stop = deque[float](maxlen=maxlen)
        self.audio_lufs = deque[float](maxlen=maxlen)
        self.audio_energy = deque[float](maxlen=maxlen)

    def push(self,
            onehot_user_play: float,
            onehot_user_stop: float,
            audio_lufs: float,
            audio_energy: float,
            *,
            falloff: float = 0.0
            ) -> None:
        """ Pushes a frame to the defined deques and applies onehot falloff if specified. """

        def adjustf(arr: deque[float], x: float) -> float:
            return max(arr[-1] * falloff, x) if len(arr) else x
        
        self.time_seconds.append(time())
        self.onehot_user_play.append(adjustf(self.onehot_user_play, onehot_user_play))
        self.onehot_user_stop.append(adjustf(self.onehot_user_stop, onehot_user_stop))
        self.audio_lufs.append(audio_lufs)
        self.audio_energy.append(audio_energy)
    
    def make_dataframe(self):
        return DataFrame({
            "time_seconds": self.time_seconds,
            "audio_lufs": self.audio_lufs,
            "audio_energy": self.audio_energy,
            "onehot_user_play": self.onehot_user_play,
            "onehot_user_stop": self.onehot_user_stop,
        })

class AudioAnalysisStream(AudioPlayerStream):
    _cfg: AnalysisPlayerOptions
    _analysis: AnalysisResult

    _rolling_data: AnalysisPlayerSeries
    # _user_actions: list[tuple[float, UserAction]]

    # Data for next frame
    _last_frame_tsec: float = 0.0
    _frame_onehot_action: UserAction = UserAction.Empty

    _thread_lock: Lock
    _thread: Thread

    def __init__(self, analysis: AnalysisResult, cfg: AnalysisPlayerOptions | None = None):
        """
        Extends AudioPlayerStream to provide time-synced audio analysis data.
        This class pipes "live" data from the analyzed music alongside user events such as playing/stopping the music in world-time instead of song-time.
        """
        super().__init__()
        self._cfg = cfg or AnalysisPlayerOptions()
        self._analysis = analysis
        # self._user_actions = []
        self._rolling_data = AnalysisPlayerSeries(self._cfg.max_points)

        self._thread_lock = Lock()
        self._thread = Thread(target=self._thread_callback)

    def _thread_callback(self) -> None:
        while True:
            if self._wants_stop: return
            if not self._stream or not self._stream.is_active(): return

            # Wait until we *should* push the next frame if we recently pushed a frame
            current_time = time()
            time_remaining = self._last_frame_tsec + self._cfg.push_timeout - current_time
            if time_remaining > 0:
                sleep(time_remaining)

            # Lock resource and push frame
            self._last_frame_tsec = current_time
            with self._thread_lock:
                self._push_frame()
    
    _get_analysis_idx__warm_start: int = 0
    """ Used by _get_analysis_window_at_time below. See comment for explanation! """

    def _get_analysis_window_at_time(self, song_time_s: float) -> tuple[int, int]:
        """ Fetches two sequential frame indices `(i_t0, i_t1)` where `t0 ≤ song_time_s` and `t1 ≥ song_time_s`. """
        time_samples = self._analysis['series']['t_sec']

        # NOTE(jadon) Silly optimization!!!
        # Usually this method is called sequentially every frame, so let's start on the previous index if it's still valid.
        i0 = self._get_analysis_idx__warm_start
        if time_samples[i0] > song_time_s:
            i0 = 0

        while i0 + 1 < len(time_samples):
            sample_s = time_samples[i0 + 1]
            if sample_s == song_time_s:
                self._get_analysis_idx__warm_start = i0
                return (i0, i0)
            if sample_s > song_time_s:
                self._get_analysis_idx__warm_start = i0
                return (i0, i0 + 1)
            i0 += 1

        return (i0, i0)

    def _push_frame(self) -> None:
        """ Call this method to push a frame to the output series. """

        # Fetch audio time
        song_s = self.get_s()
        if not song_s:
            logger.error("Attempted to push frame without song playing! WTF?")
            return;

        # Attempt to match it up to the pre-analyzed data
        i0, i1 = self._get_analysis_window_at_time(song_s)
        
        blend = 0.0
        if i0 != i1:
            t0 = self._analysis['series']['t_sec'][i0]
            t1 = self._analysis['series']['t_sec'][i1]
            blend: float = max(min((song_s - t0) / ((t1 - t0)), 1), 0)

        def b(arr: list[float]):
            return arr[i1] * blend + arr[i0] * (1.0 - blend)

        audio_lufs = b(self._analysis['series']['momentary_lufs'])
        audio_energy = b(self._analysis['series']['energy'])

        # Push frame
        self._rolling_data.push(
            float(self._frame_onehot_action == UserAction.Play),
            float(self._frame_onehot_action == UserAction.Stop),
            audio_lufs,
            audio_energy,
            falloff = self._cfg.onehot_falloff
        )

        # Reset for next frame
        self._frame_onehot_action = UserAction.Empty

    def _on_user_action(self, action: UserAction) -> None:
        """ Call this method any time the user interacts with the audio player! Data from this should be fed into onehot regression inputs. """
        # self._user_actions.append((time(), action))
        self._frame_onehot_action = action

    @override
    def open(self, wavepath: str | None = None) -> None:
        super().open(wavepath or self._analysis["meta"]["file_name"])

    @override
    def play(self, from_start: bool = False, user_action: bool = False) -> None:
        super().play(from_start)
        if not self._thread.is_alive(): self._thread.start()
        if user_action: self._on_user_action(UserAction.Play)

    @override
    def stop(self, user_action: bool = False) -> None:
        if user_action: self._on_user_action(UserAction.Stop)
        return super().stop()


# endregion Sync-stream
