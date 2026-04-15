from recorder import AudioRecorder
from time import sleep
from openwakeword import Model
import sounddevice as sd
import soundfile as sf
import io
import asyncio
import aiohttp
import numpy as np
from portals.llm_cloud import GeminiLLM
from portals.tts_cloud import Speech2Text
import base64


async def transcribe_audio_file(audio_data, samplerate, port):
    url = f"http://127.0.0.1:{port}/transcribe/"
    async with aiohttp.ClientSession() as session:
        try:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, samplerate, format="WAV")
            audio_buffer.seek(0)

            data = aiohttp.FormData()
            data.add_field(
                "file",
                audio_buffer,
                filename="speech.wav",
                content_type="audio/wav",  # "application/octet-stream",
            )
            async with session.post(url, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result["text"]
                else:
                    error_message = await resp.text()
                    return f"error: {str(error_message)}"
        except Exception as e:
            return f"error: {str(e)}"


async def fetch_transcibe(audio_data, sr):
    results = await asyncio.gather(
        transcribe_audio_file(audio_data, sr, port=6969),
        transcribe_audio_file(audio_data, sr, port=9696),
    )
    try:
        translation, lang = results[1].split("::")
    except Exception:
        translation, lang = results[1], "en"

    return {"tamil": results[0], "english": translation, "language": lang}


class Assistant:
    def __init__(
        self,
        speak_pause_wait=2,
        listen_after_speech_wait=3,
        wws_window=1,
        chunk_duration=0.512,
    ):
        self.chunk_duration = chunk_duration
        self.audcord = AudioRecorder(chunk_duration=0.512)

        self.speak_pause_wait = round(speak_pause_wait / chunk_duration)
        self.listen_after_speech_wait = round(
            listen_after_speech_wait / chunk_duration
        )

        self._wakeword_search_window = wws_window

        self._wakedet = Model()

        self._brain = GeminiLLM()
        self._speak = Speech2Text("LABS11_API")

    def detect_wakewords(self):
        frame = self.audcord.read(self._wakeword_search_window)
        int16_audio = (frame * 32767).astype(np.int16)
        preds = self._wakedet.predict(int16_audio)
        pos = any(v > 0.9 for v in preds.values())
        return pos

    def get_reply(self, duration):
        audio_data = self.audcord.read(
            duration, use_vad=False, suppress_noise=True
        )
        results = asyncio.run(
            fetch_transcibe(audio_data, self.audcord.samplerate)
        )
        print("Transcribe results", results)

        response, suc = self._brain(results)
        audio_response = self._speak(response)
        if isinstance(audio_response, dict):
            resp_audio = audio_response["audio_base_64"]
        else:
            resp_audio = audio_response.audio_base_64

        if isinstance(suc, str):
            print("TOOL FAILIURE", suc)

        audio_bytes = base64.b64decode(resp_audio)
        audio_buffer = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_buffer)
        sd.play(audio_data, sample_rate)
        sd.wait()

    def start(self):
        proc = self.audcord.run_as_daemon()
        sleep(self.chunk_duration)

        is_speaking = False
        last_speak_ts = speak_start_tm = None
        should_record = False
        time_step = 0
        while True:
            sleep(self.chunk_duration)
            speech_curr = self.audcord.is_speech()

            if speech_curr:
                if not is_speaking:
                    speak_start_tm = time_step
                    is_speaking = True
                    print("Speech start...")
                last_speak_ts = time_step

            if (
                is_speaking
                and (time_step - last_speak_ts) >= self.speak_pause_wait
            ):
                is_speaking = False
                print(
                    "speech close after pause wait exceede...",
                    time_step - last_speak_ts,
                    self.speak_pause_wait,
                )
                if should_record:
                    duration = (
                        round(
                            (time_step - speak_start_tm) * self.chunk_duration
                        )
                        # + 1  # offset 1 second
                    )
                    self.get_reply(duration)
                    last_speak_ts = time_step

            if (
                not is_speaking
                and should_record
                and (time_step - last_speak_ts)
                >= self.listen_after_speech_wait
            ):
                should_record = False
                print("Listening stoped...")

            if is_speaking and not should_record and self.detect_wakewords():
                speak_start_tm = time_step - 1
                should_record = True
                print("Active listening...")

            response = self._brain.check_reminders(is_speaking, time_step)

            time_step += 1
            if response is not None:
                print("Remainder response", response)
                audio_response = self._speak(response)
                if isinstance(audio_response, dict):
                    resp_audio = audio_response["audio_base_64"]
                else:
                    resp_audio = audio_response.audio_base_64

                audio_bytes = base64.b64decode(resp_audio)
                audio_buffer = io.BytesIO(audio_bytes)
                audio_data, sample_rate = sf.read(audio_buffer)
                sd.play(audio_data, sample_rate)
                sd.wait()

        proc.kill()


if __name__ == "__main__":
    asi = Assistant()
    asi.start()
