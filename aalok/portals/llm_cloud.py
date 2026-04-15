from .utils import BaseAPIRotator
from google.genai import Client, types
from google import genai
import json
import datetime
import io
import soundfile as sf
from twilio.rest import Client as TWClient
import os
import pendulum


class GeminiLLM(BaseAPIRotator):
    def __init__(
        self,
        key_name: str = "GEMINI_API",
        model: str = "gemini-2.0-flash",  # -lite",
    ):
        self.client = None
        self.model = model

        super().__init__(key_name, debug=True)
        self.model = os.environ["MODEL_TO_USE"]

        self.caller = TWClient(
            os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"]
        )

        with open("sys_instruction.txt", "r") as f:
            self.system_prompt = f.read()

        # load information
        with open("userinfo.json", "r") as f:
            self.user_details = json.load(f)

        with open("medicalrecords.json", "r") as f:
            self.med_records = json.load(f)

        with open("contactlist.json", "r") as f:
            self.contacts = json.load(f)

        with open("reminders.json", "r") as f:
            self.reminders = json.load(f)

        for item in self.reminders:
            item["time"] = pendulum.parse(item["time"])

        self.past_20_chats = []

    def set_client(self, api_key):
        self.client = Client(api_key=api_key)

    def _build_sys_inst(
        self,
    ):
        return self.system_prompt % (
            " - ".join(self.user_details),
            " - ".join(self.med_records),
            str(list(self.contacts.keys())),
            datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p"),
        )

    def _get_system_config(self):
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["response"],
                properties={
                    "response": genai.types.Schema(
                        type=genai.types.Type.STRING,
                    ),
                    "important": genai.types.Schema(
                        type=genai.types.Type.ARRAY,
                        items=genai.types.Schema(
                            type=genai.types.Type.STRING,
                        ),
                    ),
                    "tool": genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        properties={
                            "name": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                            "arguments": genai.types.Schema(
                                type=genai.types.Type.STRING,
                            ),
                        },
                    ),
                },
            ),
            system_instruction=[
                types.Part.from_text(text=self._build_sys_inst()),
            ],
        )
        return generate_content_config

    def _make_chat_log(self, text):
        contents = []
        for role, content in self.past_20_chats:
            contents.append(
                types.Content(
                    role=role, parts=[types.Part.from_text(text=content)]
                )
            )
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            )
        )
        return contents

    def function(self, prompt: dict[str, str]):
        if isinstance(prompt, tuple):
            audbuff = io.BytesIO()
            sf.write(audbuff, *prompt, format="WAV")  # (y, sr)
            audbuff.seek(0)
            files = [
                self.client.files.upload(
                    file=audbuff, config=dict(mime_type="audio/wav")
                )
            ]
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=files[0].uri,
                            mime_type=files[0].mime_type,
                        ),
                    ],
                ),
            ]
            llm_response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._get_system_config(),
            )
            response = llm_response.parsed

            dispay_response = response["response"]
            if len(response["important"]) > 0:
                self.user_details += response["important"]

            return dispay_response

        if any(val.startswith("error") for val in prompt.values()):
            return None

        if prompt["language"] == "en":
            usr_response = prompt["english"]
        else:
            usr_response = prompt["tamil"]

        # usr_response = f"Response In English (translated): {prompt['english']}\nResponse In Tamil (transcribed): {prompt['tamil']}\nDetected Language by model: {prompt['language']}\n\nNOTE: THE SPEECH TO TEXT MODEL IS NOT ACCURATE, YOU NEED TO COMPILE THE ABOVE INFORMATION TO INFER THE ACTUAL USER RESPONSE. IT MOSTLIKEY TO BE IN TAMIL, SO YOU CAN REPLY BACK TO THEM IN MIXTURE OF TAMIL AND ENGLISH."

        contents = self._make_chat_log(usr_response)

        llm_response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._get_system_config(),
        )
        response = llm_response.parsed
        print(response)

        dispay_response = response["response"]
        if len(response["important"]) > 0:
            self.user_details += response["important"]
            with open("userinfo.json", "w") as f:
                json.dump(self.user_details, f, indent=4)

        tool_request = response["tool"]
        tool_success = self._tool_usage(tool_request)

        self.past_20_chats.extend(
            [("user", usr_response), ("model", llm_response.text)]
        )
        if len(self.past_20_chats) > 10:
            self.past_20_chats = self.past_20_chats[-10:]

        return dispay_response, tool_success

    def __parse_args(self, args):
        return {
            k.strip(): v.strip()
            for k, v in [a.split("=") for a in args.split(";")]
        }

    def _make_call(self, name: str, message: str):
        if message.lower() == "forward":
            url = f"http://twimlets.com/forward?PhoneNumber={os.environ['FORWARD_PHONE']}"
        else:
            url = f"http://twimlets.com/message?Message={message.replace(' ', '+')}"
        call = self.caller.calls.create(
            to=self.contacts[name], from_=os.environ["TWILIO_NUMBER"], url=url
        )

        print("Made a call", call.sid)

    def check_reminders(self, is_interrupt, timestep, delay_steps=120):
        if timestep % delay_steps:
            return None

        now_date = pendulum.now()
        the_date = now_date.date()
        now_time = now_date.time()
        now_wday = now_date.day_of_week + 1
        now_rems = None
        del_ents = None
        for i, item in enumerate(self.reminders):
            if item["time"].date() > the_date:
                continue

            if item["time"].time() <= now_time:
                if not (
                    item["repeatdays"][0] == 0
                    or now_wday in item["repeatdays"]
                ):
                    continue

                if now_rems is None:
                    now_rems = []
                    del_ents = set()

                if item["repeatdays"][0] == 0:
                    del_ents.add(i)

                now_rems.append(item["message_str"])

        if now_rems is None:
            return None

        content = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=" - ".join(now_rems)),
            ],
        )

        llm_response = self.client.models.generate_content(
            model=self.model,
            contents=[content],
            config=types.GenerateContentConfig(
                system_instruction=f"""You are a warm and friendly voice, Amica, here to gently remind the user about a few things. You will receive one or more reminder notes, joined together by " - ".

Your task is to read these out clearly, in a caring and pleasant tone. You can deliver these reminders in English, Tamil, or a comfortable mix, depending on what feels most natural or if the reminder itself contains Tamil words. If there are multiple reminders, present them naturally, perhaps as a short list. Make it sound like a helpful nudge.

For example, if you receive: "Take morning medicine - Appointment with Dr. Radha at 10 AM - Call Priya"
You could say something like: "<emotion type=\"cheerful\">Hello! Just a few gentle reminders for you right now: please remember to take your morning medicine, <break time=\"0.3s\"/> you have your appointment with Dr. Radha at 10 AM, <break time=\"0.3s\"/> and also, it's time to call Priya. <break time=\"0.3s\"/> Hope you have a wonderful day!</emotion>"

Or, if you receive just one (perhaps with a Tamil name or context): "Call Paati"
You could say (in a mix, if appropriate): "<emotion type=\"empathetic\">Oru chinna reminder, Paati-kku call pannunga. <break time=\"0.3s\"/> (A small reminder, please call Grandma.)</emotion>"
Or simply in English: "<emotion type=\"empathetic\">Just a little reminder, dear, to please call Paati.</emotion>"
{"Currently, the user is being interrupted while speaking with you. So you have to tell them apalogies for this." if is_interrupt else ""}

**Current Date & Time**: {datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")} (You can use this information in your response)
Keep it brief and directly focused on delivering the reminders in a warm, human-like way, using the language that best fits."""
            ),
        )

        if del_ents:
            self.reminders = [
                self.reminders[x]
                for x in range(len(self.reminders))
                if x not in del_ents
            ]
            with open("reminders.json", "w") as f:
                json.dump(self.reminders, f, default=str)

        return llm_response.text

    def _make_reminder(self, time, date, repeatdays, message_str):
        h, m = time.split(":")
        init_ts = pendulum.now()
        if "-" in date:
            m, d, y = date.split("-")
            date_ = pendulum.date(y, m, d)
        else:
            date_ = init_ts.date()
            if "tomo" in date:
                date_ = date_.add(days=1)

        repeatdays = list(map(int, repeatdays))

        self.reminders.append(
            {
                "time": pendulum.datetime(
                    date_.year, date_.month, date_.day, int(h), int(m)
                ),
                "repeatdays": repeatdays,
                "message_str": message_str,
                "initiated_timestamp": init_ts.to_iso8601_string(),
            }
        )
        with open("reminders.json", "w") as f:
            json.dump(self.reminders, f, default=str)

        print("Successfully added a reminder", self.reminders[-1])

    def _tool_usage(self, request):
        name = request["name"]
        if "reminder" in name:
            try:
                self._make_reminder(**self.__parse_args(request["arguments"]))
            except Exception as e:
                return f"Failed to add the reminder: {e}"

            return True
        elif "call" in name:
            args = self.__parse_args(request["arguments"])
            target = None
            param = args["name"].lower()

            for key in self.contacts.keys():
                if param in key.strip().lower():
                    target = key
                    break
            if target is None:
                return f"Contact name matching {param} was not found!"

            try:
                self._make_call(target, args["message"])
            except Exception as e:
                return f"Error occured intantiating the call, couldn't complete the request: {e}"
            return True

        return False


if __name__ == "__main__":
    llm = GeminiLLM(
        "GEMINI_API",
        "gemma",
    )
    llm({})
