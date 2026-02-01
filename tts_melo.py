import torch
import numpy as np
import scipy.signal
from tts_interface import TTSInterface
import threading
import tempfile
import os
import scipy.io.wavfile

from melo.api import TTS as MeloModel

class MeloTTS(TTSInterface):
    def __init__(self, audio_io, lang="JP", device="auto"):
        self.audio_io = audio_io
        self.lock = threading.Lock()
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.lang = lang
        
        print(f"🔄 MeloTTS Loading ({lang} / {device})...")
        if MeloModel:
            try:
                # MeloTTS model load
                self.model = MeloModel(language=lang, device=device)
                self.speaker_ids = self.model.hps.data.spk2id
                print("✅ MeloTTS Ready")
            except Exception as e:
                print(f"❌ MeloTTS Init Error: {e}")
                self.model = None
        else:
            self.model = None

    def speak(self, text: str, **kwargs):
        if not self.model or not text:
            return

        # Default voice based on lang
        speaker_key = kwargs.get('voice', 'EN-US')
        if self.lang == 'JP':
            speaker_key = 'JP' 

        # Fix for HParams object not having .get()
        try:
             if hasattr(self.speaker_ids, '__dict__'):
                 spk_dict = self.speaker_ids.__dict__
             else:
                 spk_dict = vars(self.speaker_ids)
             
             if speaker_key in spk_dict:
                 speaker_id = spk_dict[speaker_key]
             else:
                 first_key = list(spk_dict.keys())[0]
                 speaker_id = spk_dict[first_key]
                 
        except Exception:
             print(f"⚠️ Speaker lookup failed for {speaker_key}, using default ID 0")
             speaker_id = 0

        speed = kwargs.get('speed', 1.5)
        print(f"🔊 MeloTTS Speaking: {text}")

        try:
            with self.lock:
                # Use 'tts_to_file' via temp file.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    temp_path = tf.name
                
                try:
                    # Generate to file
                    self.model.tts_to_file(text, speaker_id, temp_path, speed=speed)
                    
                    # Read back
                    sample_rate, audio_data = scipy.io.wavfile.read(temp_path)
                    
                    # Convert to float32 normalized
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                         audio_data = audio_data.astype(np.float32) / 2147483648.0
                    
                    if audio_data is not None and len(audio_data) > 0:
                         # Resample if needed
                        if sample_rate != self.audio_io.sample_rate:
                            num_samples = int(len(audio_data) * self.audio_io.sample_rate / sample_rate)
                            audio_data = scipy.signal.resample(audio_data, num_samples)
                        
                        self.audio_io.enqueue_output(audio_data)
                    else:
                        print("⚠️ MeloTTS generated empty audio")
                        
                finally:
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        except Exception as e:
            print(f"❌ MeloTTS Error: {e}")
