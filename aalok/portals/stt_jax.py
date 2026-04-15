from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import argparse
from whisper_jax import FlaxWhisperPipline
from contextlib import asynccontextmanager
import os


class ModelManager:
    def __init__(self):
        self.pipeline = None
        self.kwargs = None

    def load_model(self, model_name, kwargs):
        print("Loading the model...")
        print("Model mode", kwargs)
        self.pipeline = FlaxWhisperPipline(model_name, batch_size=1)
        self.kwargs = kwargs
        # print("Warming up model...")
        # self.transcribe("audios/oops.mp3")

    def transcribe(self, file):
        return self.pipeline(file, **self.kwargs)["text"]


model = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    parser = create_arg_parser()
    args = parser.parse_args()
    print(
        f"Loading & Setting the Whisper model: {args.variant} {args.model_size}"
    )
    if args.variant == "tamil":
        model_name = f"vasista22/whisper-tamil-{args.model_size}"
        task = None  # default is "transcribe"
        language = None  # default is "ta"
    else:
        model_name = f"openai/whisper-{args.model_size}"
        task = "translate"
        language = "en"

    language = args.language if args.language is not None else language
    task = args.task if args.task is not None else task

    if language == "none":
        language = None
    if task == "none":
        task = None

    model.load_model(model_name, dict(task=task, language=language))
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
        help="Whisper model size to load (base & tiny only available for english)",
    )

    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["tamil", "english"],
        help="Tamil uses fintetuned model default for transcription in tamil and English langauge server uses real model by default translation",
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
    port = 6969 if args.variant == "tamil" else 9696
    print(
        f"Serving Whisper-{args.variant}-{args.model_size} @ localhost:{port}"
    )
    uvicorn.run("stt_jax:app", host="127.0.0.1", port=port)
