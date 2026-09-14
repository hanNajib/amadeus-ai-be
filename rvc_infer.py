import os
import torch
import soundfile as sf
import fairseq.data.dictionary
from rvc_python.infer import RVCInference

# Patch safe globals PyTorch 2.6+
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
        
        # Deteksi hardware (GPU CUDA jika ada, jika tidak otomatis ke CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[RVC Engine] Device aktif: {self.device}")

        # Inisialisasi engine RVC
        self.rvc = RVCInference(device=self.device)
        if os.path.exists(self.model_path):
            self.rvc.load_model(self.model_path)
            print(f"[RVC Engine] Model Kurisu dimuat dari {self.model_path}")
        else:
            print(f"[RVC Warning] Model tidak ditemukan di {self.model_path}!")

    def convert(self, input_wav_path: str, output_wav_path: str, pitch_shift: int = 0):
        if not os.path.exists(self.model_path):
            print("[RVC Error] File model pth tidak ditemukan.")
            return input_wav_path

        try:
            # Pilihan algoritma pitch extraction:
            # - 'pm': Algoritma paling ringan khusus CPU (VPS tanpa GPU)
            # - 'rmvpe': Algoritma sangat presisi & jernih (Khusus GPU CUDA)
            f0_algorithm = "rmvpe" if self.device.startswith("cuda") else "pm"

            self.rvc.set_params(
                f0up_key=pitch_shift,
                f0method=f0_algorithm,
                index_rate=0.75,
                index_path=self.index_path if self.index_path else ""
            )
            
            # EKSEKUSI VOICE CONVERSION NEURAL NETWORK
            self.rvc.infer_file(input_wav_path, output_wav_path)
            
            if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
                return output_wav_path
            return input_wav_path

        except Exception as e:
            print(f"[RVC Inference Error] {e}")
            return input_wav_path   