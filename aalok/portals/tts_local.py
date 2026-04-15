from transformers import AutoModel
import numpy as np
import io
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
from typing import Optional

from fastapi.responses import StreamingResponse


class ModelManager:
    def __init__(self):
        self.reference_text = None
        self.reference_audio = None
        self.current_ref_id = None
        self.model = None

    def _load_reference(self, ref_id: str | None):
        if ref_id is None:
            return

        if ref_id == self.current_ref_id:
            return

        print(f"Changing reference to {ref_id}")
        self.current_ref_id = ref_id
        self.reference_audio = f"audios/speech_ref-{ref_id}.wav"

        with open(f"audios/speech_ref-{ref_id}.txt", "r") as f:
            self.reference_text = f.read()

    def load_model(self, ref_id: str = None):
        self._load_reference(ref_id)
        self.model = AutoModel.from_pretrained(
            "ai4bharat/IndicF5", trust_remote_code=True
        )

    def generate_speech(self, text: str, ref_id: str = None):
        self._load_reference(ref_id)
        audio = self.model(
            text,
            ref_audio_path=self.reference_audio,
            ref_text=self.reference_text,
        )

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        return audio, 24000


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    parser = create_arg_parser()
    args = parser.parse_args()
    assert args.ref_id is not None, "Missing initial reference"
    model_manager.load_model(args.ref_id)
    yield


app = FastAPI(lifespan=lifespan)


class InputData(BaseModel):
    text: str
    ref_id: Optional[int] = None


@app.post("/speech/")
async def generate_speech(data: InputData):
    audio_arr, sr = model_manager.generate_speech(data.text, data.ref_id)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_arr, sr, format="WAV")
    audio_buffer.seek(0)

    return StreamingResponse(audio_buffer, media_type="audio/wav")


def create_arg_parser():
    parser = argparse.ArgumentParser(description="F5 TTS FastAPI Server")
    parser.add_argument(
        "--ref_id",
        type=int,
        default=None,
        # options=[],
        help="Speech reference to use when synthesizing",
    )

    return parser


if __name__ == "__main__":
    import uvicorn

    print("Serving F5-TTS @ localhost:42069")
    uvicorn.run("tts_local:app", host="127.0.0.1", port=42069)
