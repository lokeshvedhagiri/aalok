import multiprocessing
import sounddevice as sd
import time
import numpy as np
from time import sleep
from math import ceil
from silero_vad import load_silero_vad, get_speech_timestamps
import torch
import noisereduce as nr


class SharedAudioBuffer:
    def __init__(self, max_duration, chunk_duration):
        manager = multiprocessing.Manager()
        self.buffer = manager.list()
        self.lock = manager.Lock()
        self.chunk_duration = chunk_duration
        self.maxlen = int(max_duration / chunk_duration)

    def append(self, data):
        with self.lock:
            if len(self.buffer) >= self.maxlen:
                self.buffer.pop(0)
            self.buffer.append(data)

    def get_last_chunk(self):
        with self.lock:
            return self.buffer[-1]

    def get_chunks(self, duration):
        n_chunks = ceil(duration / self.chunk_duration)
        with self.lock:
            size = len(self.buffer)
            return list(self.buffer[max(size - n_chunks, 0) :])

    def size(self):
        with self.lock:
            return len(self.buffer)


class AudioRecorder:
    def __init__(self, samplerate=16000, chunk_duration=1.0, max_duration=60):
        self.shared_buffer = SharedAudioBuffer(max_duration, chunk_duration)
        self.samplerate = samplerate
        self.chunk_size = int(self.samplerate * chunk_duration)

        self.vad = load_silero_vad(onnx=True)

    def _record_forever(self):
        def callback(indata, frames, time_info, status):
            if status:
                print("Status:", status)
            self.shared_buffer.append(indata.copy())

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,  # Mono by default
            callback=callback,
            blocksize=self.chunk_size,
        ):
            print("Recording started in daemon...")
            while True:
                time.sleep(1)

    def run_as_daemon(self):
        p = multiprocessing.Process(target=self._record_forever)
        p.daemon = True
        p.start()
        return p

    def read(self, duration: int, suppress_noise=True, use_vad=False):
        audio_chunks = self.shared_buffer.get_chunks(duration)
        audio_frame = np.vstack(audio_chunks).reshape(-1)[
            : int(duration * self.samplerate)
        ]
        if suppress_noise:
            audio_frame = nr.reduce_noise(
                audio_frame, self.samplerate, device="cpu"
            )

        if use_vad:
            speech_timestamps = get_speech_timestamps(
                torch.from_numpy(audio_frame),
                self.vad,
                return_seconds=True,
                sampling_rate=self.samplerate,
            )
            speech_segments = []

            for segment in speech_timestamps:
                start_sample = int(segment["start"] * self.samplerate)
                end_sample = int(segment["end"] * self.samplerate)
                speech_segments.append(audio_frame[start_sample:end_sample])

            if speech_segments:
                audio_frame = np.concatenate(speech_segments)

        return audio_frame

    def is_speech(self):
        audio_data = torch.from_numpy(
            self.shared_buffer.get_last_chunk().reshape(-1)
        )
        is_speech = False
        window_size_samples = 512
        for i in range(0, len(audio_data), 512):
            chunk = audio_data[i : i + window_size_samples]
            if len(chunk) < window_size_samples:
                break

            speech_prob = self.vad(chunk, self.samplerate).item()
            if speech_prob > 0.55:
                is_speech = True
                break

        self.vad.reset_states()  # reset model states after each audio
        return is_speech


if __name__ == "__main__":
    recorder = AudioRecorder(chunk_duration=0.512, max_duration=5)
    recorder_process = recorder.run_as_daemon()
    print("Main sleeping, recorder running in background...")
    sleep(1)  # min wait time till the buffer gets any data
    try:
        while True:
            sd.play(recorder.read(1), recorder.samplerate)
            sd.wait()
            print("is Speech?", recorder.is_speech())
            print(f"Audio buffer size: {recorder.shared_buffer.size()}")
    except KeyboardInterrupt:
        print("Exiting main process.")
