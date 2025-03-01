from gtts import gTTS
from pygame import mixer
from time import sleep
import io

def text_to_speech(text: str, autoplay: bool = True, output: str = None, lang: str = "id"):
    tts = gTTS(text=text, lang=lang)
    
    if output != None:
        tts.save(output)

    if autoplay:
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        play(bytes=fp)


def play(filename: str = None, bytes: io.BytesIO = None):
    mixer.init()

    # Intro
    mixer.music.load(filename="./data/sound/intro.wav")
    mixer.music.play()
    while mixer.music.get_busy():
        continue

    # Main Audio
    if bytes != None:
        mixer.music.load(bytes, "mp3")
    mixer.music.play()
    while mixer.music.get_busy():
        continue
    
    # Outro
    sleep(0.5)
    mixer.music.load(filename="./data/sound/outro.wav")
    mixer.music.play()
    while mixer.music.get_busy():
        continue