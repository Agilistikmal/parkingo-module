import threading
import time
from typing import Literal
from playsound3 import playsound

LAST_PLATE_SOUND = {}


def play_sound(
    sound: Literal["valid", "invalid", "guest"] = "guest",
    camera_source: str = None,
    plate_number: str = None,
):
    global LAST_PLATE_SOUND

    last_camera_plate_sound = LAST_PLATE_SOUND.get(camera_source)
    now = time.time()
    # if the last plate sound is the same as the current plate sound,
    # and the timestamp is less than 1 minute, return
    if last_camera_plate_sound and now - last_camera_plate_sound.get("timestamp") < 60:
        return
    elif (
        last_camera_plate_sound and now - last_camera_plate_sound.get("timestamp") > 60
    ):
        if sound == "valid" or sound == "guest":
            return

    LAST_PLATE_SOUND[camera_source] = {
        "camera_source": camera_source,
        "plate_number": plate_number,
        "sound": sound,
        "timestamp": time.time(),
    }

    threading.Timer(60 * 60, remove_last_plate_sound, args=(camera_source,)).start()

    # Run in non blocking thread
    threading.Thread(target=play_sound_thread, args=(sound,)).start()


def play_sound_thread(sound: Literal["valid", "invalid", "guest"] = "guest"):
    if sound == "valid":
        playsound("./data/sound/intro.wav", block=False)
        playsound("./data/sound/welcome_reserved.mp3")
        playsound("./data/sound/outro.wav")
    elif sound == "invalid":
        playsound("./data/sound/alert_volume_down.mp3", block=False)
        playsound("./data/sound/error_reserved.mp3")
        playsound("./data/sound/outro.wav")
    elif sound == "guest":
        playsound("./data/sound/intro.wav")
        playsound("./data/sound/outro.wav")


def remove_last_plate_sound(camera_source: str):
    global LAST_PLATE_SOUND
    LAST_PLATE_SOUND.pop(camera_source)


if __name__ == "__main__":
    play_sound("guest")
    time.sleep(1)
    play_sound("valid")
    time.sleep(1)
    play_sound("invalid")
