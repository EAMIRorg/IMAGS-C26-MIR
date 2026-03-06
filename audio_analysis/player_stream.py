from logging import getLogger
from collections.abc import Mapping

import wave as wavelib
import pyaudio

logger = getLogger("AudioPlayerStream")

class AudioPlayerStream:
    # _analysis: AnalysisResult
    _wants_stop: bool
    _time_start: float
    _audio: pyaudio.PyAudio | None = None
    _filepath: str | None = None
    _file: wavelib.Wave_read | None = None
    _stream: pyaudio.Stream | None = None

    def __init__(self):
        """
        A class for quickly streaming wave files to the system audio device.

        ## Example

        ```py
        # Create a new player and load the audio.
        player = AudioPlayerStream()
        player.open("./somefile.wav")

        # Play the file asynchronously.
        player.play()

        # Wait until finished.
        while player.is_playing()
            time.sleep(0.1)

        # Free the file and other resources.
        player.close()
        ```
        """

        # self._analysis = analysis
        self._wants_stop = False
        self._time_start = 0.0
    
    # def __enter__(self):
    def open(self, wavepath: str | None = None):
        assert not self._audio, "Cannot open an AudioPlayerStream twice!"
        self._filepath = wavepath
        self._init_file()
        self._audio = pyaudio.PyAudio()
        # return self

    # def __exit__(self, _exc_type, _exc_value, _exc_trace):
    def close(self):
        """ Closes the stream, the file, and the PortAudio instance. """
        assert self._audio, "Cannot close an AudioPlayerStream that is not open!"
        self._deinit_stream()
        self._deinit_file()
        self._audio.terminate()
        self._audio = None

    def is_playing(self) -> bool:
        return self._stream != None and self._stream.is_active()

    def _stream_callback(
            self,
            _in_data: bytes | None,
            frame_count: int,
            _time_info: Mapping[str, float],
            _status: int
            ) -> tuple[bytes | None, int]:
        """ Called by PortAudio whenever the stream wants to pull more data. """

        if self._wants_stop:
            return (bytes(0), pyaudio.paAbort)

        assert self._file
        frames = self._file.readframes(frame_count)
        return (frames, pyaudio.paContinue)

    def _init_stream(self) -> None:
        """ Starts pulling data from the current wave file. """
        assert self._audio, "AudioPlayerStream.open must be called before audio can play!"
        
        if not self._file:
            raise Exception("AudioPlayerStream: Attempted to start audio stream without file loaded!")
        if self._stream:
            logger.warning("Unexpected _init_stream call with active stream! De-initing previous stream...")
            self._deinit_stream()

        self._stream = self._audio.open(
            format=self._audio.get_format_from_width(self._file.getsampwidth()),
            channels=self._file.getnchannels(),
            rate=self._file.getframerate(),
            stream_callback=self._stream_callback,
            output=True,
        )
    
    def _deinit_stream(self) -> None:
        """ Closes the audio stream, stopping audio. """
        if self._stream:
            if self._stream.is_active():
                self._stream.stop_stream();
            self._stream.close();
            self._stream = None

    def _init_file(self) -> None:
        """ Loads a wave from this stream's path. """
        assert self._filepath, "A filepath must be specified to open at least once!"
        if self._file:
            logger.warning("Unexpected _init_file call with active file! De-initing previous file...")
            self._deinit_file()
        self._file = wavelib.open(self._filepath, 'rb')
    
    def _deinit_file(self) -> None:
        """ Unloads the currently-loaded wave. """
        if self._file:
            self._file.close()
            self._file = None

    def play(self, from_start: bool = False) -> None:
        """ Begins playing the active wave. """
        assert self._file

        self._deinit_stream()

        if from_start:
            self._file.rewind()

        self._wants_stop = False
        self._init_stream()

    def stop(self) -> None:
        self._wants_stop = True
        if self._stream:
            self._stream.stop_stream()

    def seek_frame(self, frame: int = 0) -> None:
        if not self._file: return logger.error("Cannot seek when no file is loaded!")
        self._file.setpos(frame)

    def seek_s(self, time: float = 0.0) -> None:
        if not self._file: return logger.error("Cannot seek when no file is loaded!")
        self._file.setpos(round(self._file.getframerate() * time))

    def get_frame(self) -> int | None:
        return self._file.tell() if self._file else None
    
    def get_s(self) -> float | None:
        return (self._file.tell() / self._file.getframerate()) if self._file else None
