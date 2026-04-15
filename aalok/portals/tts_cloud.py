from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from .utils import BaseAPIRotator
import base64

load_dotenv()


class Speech2Text(BaseAPIRotator):
    def __init__(self, key_name: str):
        self.client = None
        super().__init__(key_name, debug=True)

    def set_client(self, api_key):
        self.client = ElevenLabs(api_key=api_key)

    def function(self, text: str):
        """Generate speech"""

        if text is None:
            with open("audios/oops.mp3", "rb") as mp3_file:
                encoded_string = base64.b64encode(mp3_file.read()).decode(
                    "utf-8"
                )
            return {"fallback": "text None", "audio_base_64": encoded_string}

        response = self.client.text_to_speech.convert_with_timestamps(
            voice_id="6p0P6gezgvY1v6xbLzmU",
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_flash_v2_5",
            apply_language_text_normalization=False,
            apply_text_normalization="off",
        )
        # with open(f"{time()}.mp3", "wb") as audio_file:
        #     audio_file.write(base64.b64decode(response.audio_base_64))
        return response


if __name__ == "__main__":
    spr = Speech2Text("LABS11_API")
    text = "There should be everything anything to you."
    chunks = spr(text)
