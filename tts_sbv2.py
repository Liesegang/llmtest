import requests
import numpy as np
import scipy.io.wavfile
import scipy.signal
import io
import threading
import urllib.parse
from tts_interface import TTSInterface

class SBV2TTS(TTSInterface):
    def __init__(self, audio_io, api_url="http://127.0.0.1:5000", model_id=8, style="happy", style_weight=1.0):
        self.audio_io = audio_io
        self.api_url = api_url.rstrip("/")
        self.model_id = model_id
        self.style = style
        self.style_weight = style_weight
        self.lock = threading.Lock()
        print(f"🔄 SBV2TTS API Setup: {self.api_url} (Model={model_id})")

    def speak(self, text: str, **kwargs):
        if not text:
            return

        print(f"🔊 SBV2 Speaking: {text}")
        
        # Override params from kwargs if provided
        model_id = kwargs.get('model_id', self.model_id)
        style = kwargs.get('style', self.style)
        # weight = kwargs.get('weight', self.style_weight) # API might vary on param name

        # Construct Query
        # Common Style-Bert-VITS2 API params:
        # text, model_id, speaker_id (sometimes), style, style_weight
        # We assume the "Voicevox compatible" or "Simple generic" API?
        # Let's try the common endpoint for local servers: /voice
        # params: text, model_id, speaker_id, sdp_ratio, noise, noisew, length, language, auto_split, split_interval, assist_text, assist_text_weight, style, style_weight
        
        params = {
            "text": text,
            "model_id": model_id,
            "speaker_id": 0, # Default speaker
            "style": style,
            "style_weight": self.style_weight,
            "language": "JP", # Force JP usually
            "encoding": "utf-8" # Make sure text is handled right
        }

        try:
            with self.lock:
                # Request to API
                # Endpoint might be /voice or /synthesis. 
                # Let's try /voice which is common for some wrappers. 
                # OR user might be using the official "server_fastapi.py"?
                # Official endpoint: GET /voice
                
                url = f"{self.api_url}/voice"
                
                response = requests.get(url, params=params)
                
                if response.status_code != 200:
                    print(f"⚠️ SBV2 API Error {response.status_code}: {response.text}")
                    return

                # Response content should be WAV bytes
                audio_content = response.content
                
                # Decode WAV
                with io.BytesIO(audio_content) as bio:
                    sample_rate, audio_data = scipy.io.wavfile.read(bio)
                
                # Convert to float32
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                
                # Resample
                if len(audio_data) > 0:
                    if sample_rate != self.audio_io.sample_rate:
                        num_samples = int(len(audio_data) * self.audio_io.sample_rate / sample_rate)
                        audio_data = scipy.signal.resample(audio_data, num_samples)
                    
                    self.audio_io.enqueue_output(audio_data)
                else:
                    print("⚠️ SBV2 generated empty audio")

        except Exception as e:
            print(f"❌ SBV2 Error: {e}")
            print(f"   (Is the API server running at {self.api_url}?)")
