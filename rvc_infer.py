import os
import soundfile as sf
from rvc_python.infer import infer_file

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

    def convert(self, input_wav_path: str, output_wav_path: str, pitch_shift: int = 0):
        if not os.path.exists(self.model_path):
            print("[RVC Error] File model pth tidak ditemukan, menggunakan audio asli.")
            return input_wav_path

        try:
            # Eksekusi RVC Inference Pipeline
            infer_file(
                input_path=input_wav_path,
                model_path=self.model_path,
                index_path=self.index_path if self.index_path else "",
                device="cpu",               # Server Dokploy CPU
                f0_method="pm",             # 'pm' sangat cepat dan ringan di CPU VPS
                f0_up_key=pitch_shift,      # Set 0 jika base suara sudah wanita
                index_rate=0.75,            # Kemiripan karakter Kurisu 75%
                output_path=output_wav_path
            )
            return output_wav_path

        except Exception as e:
            print(f"[RVC Inference Error] {e}")
            return input_wav_path