import os
import soundfile as sf
import torch
import fairseq.data.dictionary
from rvc_python.infer import RVCInference

# Fix untuk PyTorch 2.6+ di mana weights_only=True memblokir fairseq Dictionary
try:
    torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])
except AttributeError:
    pass

class KurisuVoiceConverter:
    def __init__(
        self, 
        model_path="weights/KurisuRVCv147.pth", 
        index_path="weights/KurisuRVCv147.index"
    ):
        self.model_path = model_path
        self.index_path = index_path if os.path.exists(index_path) else None
        self.opt_path = "temp_rvc_out.wav"
        
        if os.path.exists(self.model_path):
            print(f"[RVC] Model Kurisu terdeteksi di {self.model_path}")
        else:
            print(f"[RVC Warning] Model tidak ditemukan di {self.model_path}!")

        # Initialize the RVC Inference engine
        self.rvc = RVCInference(device="cpu")
        if os.path.exists(self.model_path):
            self.rvc.load_model(self.model_path)

    def convert(self, input_wav_path: str, output_wav_path: str, pitch_shift: int = 0):
        if not os.path.exists(self.model_path):
            print("[RVC Error] File model pth tidak ditemukan, menggunakan audio asli.")
            return input_wav_path

        try:
            # Set parameters for RVC Inference (menggunakan nama parameter yang didukung rvc_python v0.1.5)
            self.rvc.set_params(
                f0up_key=pitch_shift,
                f0method="pm",
                index_rate=0.75,
                index_path=self.index_path if self.index_path else ""
            )
            
            # Execute Inference Pipeline
            self.rvc.infer_file(input_wav_path, output_wav_path)
            
            # Validasi apakah file berhasil dibuat dan ukurannya lebih dari 0 bytes
            if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
                return output_wav_path
            else:
                print(f"[RVC Error] Output file {output_wav_path} kosong (0 bytes) atau gagal dibuat.")
                return input_wav_path

        except Exception as e:
            print(f"[RVC Inference Error] {e}")
            return input_wav_path