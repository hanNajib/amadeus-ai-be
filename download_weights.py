import os
import urllib.request

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

FILES = {
    "KurisuRVCv147.pth": "https://huggingface.co/FrancescoCaracciolo/Kurisu-RVC/resolve/main/KurisuRVCv147.pth",
    "KurisuRVCv147.index": "https://huggingface.co/FrancescoCaracciolo/Kurisu-RVC/resolve/main/KurisuRVCv147.index",
    "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
}

def download_all():
    for filename, url in FILES.items():
        filepath = os.path.join(WEIGHTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[Downloading] {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"[Done] {filename}")
        else:
            print(f"[Skipped] {filename} already exists.")

if __name__ == "__main__":
    download_all()