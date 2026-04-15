from dotenv import load_dotenv
import traceback
import os

load_dotenv()


class BaseAPIRotator:
    def __init__(
        self, var_name: str, debug: bool = False, max_tries: int = None
    ):
        keys = os.getenv(var_name).split(",")
        if len(keys) == 1:
            print("Can't rotate keys... might face rate limit errors.")

        self.__registery = {
            f"k{i}": dict(key=key, count=0) for i, key in enumerate(keys)
        }
        self.__index = "k0"

        self._debug = debug
        self._try_count = 0
        self._max_tries = max_tries if max_tries is not None else len(keys)

        self.set_client(self.__registery[self.__index]["key"])

    def __rotate_api(self):
        if self._try_count >= self._max_tries or len(self.__registery) == 1:
            return False

        nxt_key = None
        curr_count = self.__registery[self.__index]["count"]
        for kid, reg in self.__registery.items():
            if reg["count"] <= curr_count and kid != self.__index:
                nxt_key = kid
                if reg["count"] < curr_count:
                    break

        self.__index = nxt_key
        self.set_client(self.__registery[self.__index]["key"])
        self._try_count += 1
        return True

    def set_client(self, api_key):
        raise NotImplementedError("Missing client set/reseter")

    def function(self, *args, **kwargs):
        raise NotImplementedError("Can't find the main function")

    def __call__(self, *args, **kwargs):
        rval = None
        try:
            rval = self.function(*args, **kwargs)
        except Exception as e:
            if self._debug:
                print("Exception occured", e)
                traceback.print_exc()
            if self.__rotate_api():
                rval = self.function(*args, **kwargs)

        if self._debug:
            print(
                f"Took {self._try_count} rotations to access api, isSuccess?[{rval is not None}] "
            )

        self._try_count = 0  # reset the try loop count
        self.__registery[self.__index]["count"] += int(rval is not None)
        return rval

    def print_count(self):
        print(f"Current key: {self.__index}")
        for k, v in self.__registery.items():
            print(f"KeyID-{k}: {v['count']}")
