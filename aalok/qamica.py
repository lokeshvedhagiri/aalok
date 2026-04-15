# assistant_logic.py (your existing Assistant and related imports, modified)
import threading
import sounddevice as sd
import soundfile as sf
import io
import asyncio
import aiohttp
import numpy as np
import base64
from time import sleep  # Added time for timestamping

# Assuming these are in the same directory or PYTHONPATH
from recorder import AudioRecorder
from openwakeword import Model
from portals.llm_cloud import GeminiLLM  # Make sure this path is correct
from portals.tts_cloud import Speech2Text  # Make sure this path is correct

from PyQt6.QtCore import QObject, pyqtSignal  # For Qt integration


# --- Async Transcription Functions (from your code) ---
async def transcribe_audio_file(audio_data, samplerate, port):
    # ... (your existing transcribe_audio_file code) ...
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
                content_type="audio/wav",
            )
            async with session.post(url, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result["text"]
                else:
                    error_message = await resp.text()
                    print(
                        f"Transcription error on port {port}: {error_message}"
                    )
                    return f"error: {str(error_message)}"
        except Exception as e:
            print(f"Exception during transcription on port {port}: {e}")
            return f"error: {str(e)}"


async def fetch_transcibe(audio_data, sr):
    # ... (your existing fetch_transcibe code) ...
    results = await asyncio.gather(
        transcribe_audio_file(audio_data, sr, port=6969),
        transcribe_audio_file(audio_data, sr, port=9696),
    )
    try:
        # Ensure results[1] is a string before trying to split
        if isinstance(results[1], str):
            parts = results[1].split("::")
            translation = parts[0]
            lang = (
                parts[1] if len(parts) > 1 else "en"
            )  # Default lang if not present
        else:  # Handle case where transcription failed or returned unexpected format
            translation = str(
                results[1]
            )  # Convert to string, might be an error message
            lang = "en"  # Default
        print(results[1])
    except Exception as e:
        print(f"Error processing translation result: {results[1]} -> {e}")
        translation, lang = str(results[1]), "en"

    return {"tamil": results[0], "english": translation, "language": lang}


class Assistant(QObject):  # Inherit from QObject
    # --- GUI Signals ---
    # (Status updates, state changes)
    status_update_signal = pyqtSignal(str)  # General status messages
    listening_state_changed_signal = pyqtSignal(
        bool
    )  # True if listening, False otherwise
    processing_update_signal = pyqtSignal(
        str
    )  # "Transcribing...", "Thinking..."
    user_text_ready_signal = pyqtSignal(
        dict
    )  # {"english": "text", "tamil": "text", "language": "lang"}
    ai_text_ready_signal = pyqtSignal(str)  # AI's response text
    ai_is_speaking_signal = pyqtSignal(
        bool
    )  # True when AI starts speaking, False when done
    chat_history_updated_signal = pyqtSignal(list)  # Full chat history
    audio_chunk_for_waveform_signal = pyqtSignal(
        np.ndarray
    )  # For live waveform

    def __init__(
        self,
        speak_pause_wait=2,
        listen_after_speech_wait=3,
        wws_window=1,  # seconds for wake word detection window
        chunk_duration=0.512,  # seconds
        parent=None,  # For QObject
    ):
        super().__init__(parent)  # QObject.__init__
        self.chunk_duration = chunk_duration
        self.samplerate = 16000  # Define samplerate here or get from audcord
        self.audcord = AudioRecorder(
            samplerate=self.samplerate, chunk_duration=self.chunk_duration
        )

        self.speak_pause_wait_steps = round(speak_pause_wait / chunk_duration)
        self.listen_after_speech_wait_steps = round(
            listen_after_speech_wait / chunk_duration
        )
        self._wakeword_search_window_duration = wws_window

        self._wakedet = (
            Model()
        )  # Consider pre-loading models if they are large

        self._brain = GeminiLLM()
        self._speak = Speech2Text(
            "LABS11_API"
        )  # Ensure your API key/config is correct

        self.running = False
        self.audcord_proc = None
        self.chat_history = []

        self._is_speaking_internal = False  # VAD based
        self._should_record_internal = (
            False  # Master recording flag (wake word or manual)
        )
        self._last_speak_timestep = 0
        self._speak_start_timestep = 0
        self._time_step = 0

        # For manual GUI control
        self.manual_listen_request = False
        self.force_process_request = False

        self._no_stt = False

    def _emit_status(self, message):
        print(f"ASSISTANT: {message}")  # Keep console logs for debugging
        self.status_update_signal.emit(message)

    def detect_wakewords(self):
        # Reads a longer chunk for wake word detection
        frame = self.audcord.read(
            self._wakeword_search_window_duration,
            use_vad=False,
            suppress_noise=True,
        )  # Raw audio for WW
        if frame is None or len(frame) == 0:
            return False
        int16_audio = (frame * 32767).astype(np.int16)
        try:
            preds = self._wakedet.predict(int16_audio)
            pos = any(
                v > 0.8 for v in preds.values()
            )  # Adjust threshold as needed
            return pos
        except Exception as e:
            print(f"Wake word detection error: {e}")
            return False

    def _handle_get_reply(self, duration):
        self.processing_update_signal.emit("Recording audio...")

        self._emit_status(
            f"Capturing {duration:.2f}s of audio for processing."
        )
        audio_data = self.audcord.read(
            duration,
            use_vad=False,
            suppress_noise=True,
        )

        if (
            audio_data is None or len(audio_data) < self.samplerate * 0.1
        ):  # Need at least 0.1s
            self._emit_status(
                "No valid audio data captured after VAD for STT."
            )
            self.processing_update_signal.emit("No speech detected.")
            # QTimer.singleShot(1000, lambda: self.processing_update_signal.emit("")) # Clear after a delay
            return

        if not self._no_stt:
            self.processing_update_signal.emit("Transcribing...")
            results = asyncio.run(
                fetch_transcibe(audio_data, self.audcord.samplerate)
            )
            self._emit_status(f"Transcription results: {results}")

            user_english_text = results.get("english", "Transcription error")

            if "error" in user_english_text.lower():
                self._emit_status(f"STT Error: {user_english_text}")
                self.processing_update_signal.emit("STT Error. Try again.")
                self.user_text_ready_signal.emit(
                    {
                        "english": user_english_text,
                        "tamil": results.get("tamil", ""),
                        "language": results.get("language", "en"),
                    }
                )
                # QTimer.singleShot(2000, lambda: self.processing_update_signal.emit(""))
                return
            if results["language"] == "en":
                user_text = results["english"]
            else:
                user_text = results["tamil"]
            self.user_text_ready_signal.emit(results)  # Emit full dict
            self.chat_history.append(("user", user_text))
            self.chat_history_updated_signal.emit(list(self.chat_history))
        else:
            self.user_text_ready_signal.emit(
                {
                    "english": "Passing audio data directly to LLM.",
                    "language": "en",
                    "tamil": "",
                }
            )  # Emit full dict
            results = (audio_data, self.samplerate)

        self.processing_update_signal.emit("Thinking...")
        try:
            response_text, tool_success = self._brain(
                results
            )  # Pass the whole dict if your LLM needs it
            if isinstance(tool_success, str):
                self._emit_status(f"Tool Usage error: {tool_success}")
        except Exception as e:
            self._emit_status(f"LLM Error: {e}")
            response_text = "Sorry, I had trouble processing that."
            self.processing_update_signal.emit("LLM Error.")
            # QTimer.singleShot(2000, lambda: self.processing_update_signal.emit(""))

        self.ai_text_ready_signal.emit(response_text)
        self.chat_history.append(("ai", response_text))
        self.chat_history_updated_signal.emit(list(self.chat_history))

        self.processing_update_signal.emit("Synthesizing speech...")
        try:
            audio_response = self._speak(response_text)
            if isinstance(audio_response, dict):
                resp_audio_b64 = audio_response["audio_base_64"]
            else:
                resp_audio_b64 = audio_response.audio_base_64

            audio_bytes = base64.b64decode(resp_audio_b64)
            audio_buffer = io.BytesIO(audio_bytes)
            tts_audio_data, tts_sample_rate = sf.read(audio_buffer)

            self.processing_update_signal.emit("Speaking...")  # Or clear it

            def play_audio():
                sd.play(tts_audio_data, tts_sample_rate)
                sd.wait()  # Will block this thread, not the main one

            pth = threading.Thread(target=play_audio)
            pth.start()

        except Exception as e:
            self._emit_status(f"TTS Error or playback error: {e}")
            self.processing_update_signal.emit("TTS Error.")

        self.processing_update_signal.emit("")
        return pth

    def run_main_loop(self):  # This will be run in QThread
        self.running = True
        self.audcord_proc = self.audcord.run_as_daemon()
        self._emit_status("Audio recording daemon started.")
        sleep(self.chunk_duration)  # Wait for buffer to fill a bit

        self._time_step = 0
        self._last_speak_timestep = 0  # Initialize here
        self._speak_start_timestep = 0  # Initialize here
        try:
            while self.running:
                sleep(self.chunk_duration)
                current_speech_vad = self.audcord.is_speech()

                # --- State Machine Logic ---
                if current_speech_vad:
                    if not self._is_speaking_internal:  # Speech just started
                        self._is_speaking_internal = True
                        self._speak_start_timestep = self._time_step
                    self._last_speak_timestep = self._time_step

                # Condition to end a speech segment and process if recording
                if (
                    self._is_speaking_internal
                    and (self._time_step - self._last_speak_timestep)
                    >= self.speak_pause_wait_steps
                ):
                    self._is_speaking_internal = (
                        False  # Speech segment ended due to pause
                    )
                    self._emit_status(
                        f"Speech segment ended due to pause. Still recording: {self._should_record_internal}"
                    )
                    if self._should_record_internal:
                        # Speech segment ended, and we were recording. Process it.
                        duration_to_process_steps = (
                            round(
                                (self._time_step - self._speak_start_timestep)
                                * self.chunk_duration
                            )
                            + 1
                        )
                        self._emit_status(
                            f"Processing captured audio. Duration seconds: {duration_to_process_steps}"
                        )
                        pth = self._handle_get_reply(duration_to_process_steps)
                        self.ai_is_speaking_signal.emit(True)
                        if pth:
                            pth.join()
                        self.ai_is_speaking_signal.emit(False)
                        self.listening_state_changed_signal.emit(True)

                        # Not ideal in crowded area
                        self._should_record_internal = (
                            False  # Reset recording state after processing
                        )
                        self.listening_state_changed_signal.emit(False)
                        self._last_speak_timestep = (
                            self._time_step
                        )  # Update to prevent immediate re-trigger

                if (
                    not self._is_speaking_internal
                    and self._should_record_internal
                    and (self._time_step - self._last_speak_timestep)
                    >= self.listen_after_speech_wait_steps
                ):
                    self._emit_status(
                        "Stopping recording session due to extended silence."
                    )
                    self._should_record_internal = False
                    self.listening_state_changed_signal.emit(False)

                if (
                    self._is_speaking_internal
                    and not self._should_record_internal
                    and self.detect_wakewords()
                ):
                    self._should_record_internal = True
                    self.listening_state_changed_signal.emit(True)
                    self._speak_start_timestep = self._time_step
                    self._emit_status("Activated by wakeword")

                response = self._brain.check_reminders(
                    self._is_speaking_internal,
                    self._time_step,
                    delay_steps=120,  # a minute delay for .512s per step
                )

                self._time_step += 1
                if response is not None:
                    try:
                        self.listening_state_changed_signal.emit(False)
                        reminder_text = "Triggered reminder(s) set by user"
                        self.user_text_ready_signal.emit(
                            dict(
                                language="en",
                                english=reminder_text,
                                tamil="",
                            )
                        )  # Emit full dict
                        self.chat_history.append(("user", reminder_text))

                        self.processing_update_signal.emit(
                            "Synthesizing speech..."
                        )
                        audio_response = self._speak(response)
                        if isinstance(audio_response, dict):
                            resp_audio_b64 = audio_response["audio_base_64"]
                        else:
                            resp_audio_b64 = audio_response.audio_base_64

                        audio_bytes = base64.b64decode(resp_audio_b64)
                        audio_buffer = io.BytesIO(audio_bytes)
                        tts_audio_data, tts_sample_rate = sf.read(audio_buffer)
                        self.processing_update_signal.emit("Speaking...")

                        def play_audio():
                            sd.play(tts_audio_data, tts_sample_rate)
                            sd.wait()  # Will block this thread, not the main one

                        pth = threading.Thread(target=play_audio)
                        pth.start()
                    except Exception as e:
                        pth = None
                        self._emit_status(f"Error occured when reminding: {e}")

                    self.ai_is_speaking_signal.emit(True)
                    if pth:
                        pth.join()
                    self.ai_is_speaking_signal.emit(False)

                    # continue conversation
                    if self._is_speaking_internal:
                        self.listening_state_changed_signal.emit(True)
                    else:
                        self.processing_update_signal.emit("")
        finally:
            if self.audcord_proc:
                self._emit_status("Stopping audio recording daemon...")
                self.audcord_proc.kill()
                self.audcord_proc.join(timeout=1)
                if self.audcord_proc.is_alive():
                    self._emit_status(
                        "Audio daemon did not terminate cleanly."
                    )
                else:
                    self._emit_status("Audio daemon stopped.")
            self._emit_status("Assistant main loop finished.")

    # --- Methods for GUI to call ---
    def request_manual_listen(self):
        """Called by GUI to start listening manually."""
        if not self._should_record_internal:
            self._emit_status("GUI: Manual listen requested.")
            self.manual_listen_request = True
        else:
            self._emit_status(
                "GUI: Manual listen requested, but already in a recording session."
            )

    def request_force_process(self):
        """Called by GUI to stop current recording and process immediately."""
        if self._should_record_internal:
            self._emit_status("GUI: Force process requested.")
            self.force_process_request = True
        else:
            self._emit_status(
                "GUI: Force process requested, but not currently recording."
            )

    def stop_assistant_processing(self):
        self._emit_status("Assistant processing stop requested.")
        self.running = False
