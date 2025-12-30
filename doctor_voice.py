import os
from dotenv import load_dotenv
import elevenlabs
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from gtts import gTTS

# Load environment variables from .env file
load_dotenv()

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Setup Text to Speech-TTS-model with gTTS 
def text_to_speech_with_gtts(input_text, output_filepath):
    language="en"
    audioobj=gTTS(
        text=input_text,
        lang = language,
        slow= False
    )
    audioobj.save(output_filepath)
    return output_filepath # Return path for Gradio

# Setup Text to Speech-TTS-model with ElevenLabs
def text_to_speech_with_elevenlabs(input_text, output_filepath):
    if not ELEVENLABS_API_KEY:
        print("⚠️ ELEVENLABS_API_KEY not found! Falling back to gTTS...")
        return text_to_speech_with_gtts(input_text, output_filepath)
    
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio=client.text_to_speech.convert(
            text=input_text,
            voice_id="9BWtsMINqrJLrRacOk9x", #Aria
            output_format="mp3_22050_32",
            model_id="eleven_multilingual_v2" # Changed to Multilingual model for better language support
        )
        save(audio, output_filepath)
        return output_filepath # Return path for Gradio
    except Exception as e:
        print(f"⚠️ ElevenLabs TTS error: {e}")
        print("   Falling back to gTTS...")
        return text_to_speech_with_gtts(input_text, output_filepath)