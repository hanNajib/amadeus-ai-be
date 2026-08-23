import os
import torch
import soundfile as sf
import librosa
import numpy as np

class KurisuVoiceConverter:
    def __init__(
        self, 
        model_path="weights/KurisuRVCv147.pth", 
        index_path="weights/KurisuRVCv147.index",
        hubert_path="weights/hubert_base.pt"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.index_path = index_path
        self.hubert_path = hubert_path
        self.tgt_sr = 40000
        self._load_models()

    def _load_models(self):
        if os.path.exists(self.hubert_path):
            print(f"[RVC] HuBERT base terdeteksi di {self.hubert_path}")
        else:
            print(f"[RVC Warning] HuBERT base tidak ditemukan di {self.hubert_path}!")

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.tgt_sr = checkpoint.get("sr", 40000)
            print(f"[RVC] Model Kurisu dimuat (Target SR: {self.tgt_sr})")
        else:
            print(f"[RVC Warning] Model Kurisu tidak ditemukan di {self.model_path}!")

    def convert(self, input_wav_path: str, output_wav_path: str, pitch_shift: int = 0):
        if not os.path.exists(self.model_path):
            return input_wav_path

        try:
            audio, _ = librosa.load(input_wav_path, sr=self.tgt_sr)
            
            if pitch_shift != 0:
                audio = librosa.effects.pitch_shift(audio, sr=self.tgt_sr, n_steps=pitch_shift)

            sf.write(output_wav_path, audio, self.tgt_sr)
            return output_wav_path

        except Exception as e:
            print(f"[RVC Conversion Error] {e}")
            return input_wav_path