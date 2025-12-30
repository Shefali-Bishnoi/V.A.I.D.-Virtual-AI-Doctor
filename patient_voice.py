#Step1 : Setup Audio recorder(ffmpeg & portaudio)
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path,timeout=20,phrase_time_limit=None):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0

    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise")
            recognizer.adjust_for_ambient_noise(source,duration=1)
            logging.info("Start speaking now...")
            audio_data = recognizer.listen(source,timeout=timeout,phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path,format="mp3",bitrate="128k")
            logging.info(f"Audio saved to {file_path}")
            
    except Exception as e:
        logging.error(f"An error occured: {e}")   

#Step2 : Setup Speech to text-STT-model for transcription
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# UPDATED: Added 'language' parameter (defaults to "en")
def transcribe_with_groq(GROQ_API_KEY, audio_filepath, stt_model="whisper-large-v3", language="en"):
    client = Groq(api_key=GROQ_API_KEY)
    audio_file = open(audio_filepath,"rb")
    transcription = client.audio.transcriptions.create(
        model = stt_model,
        file=audio_file,
        language = language # Uses the selected language
    )
    return transcription.text