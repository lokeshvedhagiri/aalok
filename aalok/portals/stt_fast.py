from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import argparse

from faster_whisper import WhisperModel
from contextlib import asynccontextmanager
import os


class ModelManager:
    def __init__(self):
        self.pipeline = None
        self.kwargs = dict(
            beam_size=5,
            best_of=5,
            task="translate",
            vad_filter=False,
        )

    def load_model(self, model_name, kwargs):
        print("Model mode", kwargs)
        self.pipeline = WhisperModel(model_name)
        self.kwargs.update(kwargs)

    def transcribe(self, file):
        seg, info = self.pipeline.transcribe(file, **self.kwargs)
        text = []
        for seg in seg:
            text.append(seg.text)
        return " ".join(text) + f"::{info.language}"


model = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    parser = create_arg_parser()
    args = parser.parse_args()
    print(f"Loading & Setting the Whisper {args.model_size}")

    language = args.language
    task = args.task

    if language == "none":
        language = None
    if task == "none":
        task = None

    if not task:
        task = "translate"

    model.load_model(f"{args.model_size}", dict(task=task, language=language))
    yield
    print("Shutting down model...")


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
            temp.write(await file.read())
            temp_path = temp.name
        result = model.transcribe(temp_path)
        return {"text": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Whisper FastAPI Server")
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "tiny", "base"],
        help="Whisper model size to load",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=["translate", "transcribe", "none"],
        help="Pick a task to do",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["en", "ta", "none"],
        help="Target language",
    )
    return parser


if __name__ == "__main__":
    import uvicorn

    parser = create_arg_parser()
    args = parser.parse_args()
    print(f"Serving Whisper-{args.model_size} @ localhost:9696")
    uvicorn.run("stt_fast:app", host="127.0.0.1", port=9696)
