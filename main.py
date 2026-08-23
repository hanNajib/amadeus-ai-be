import os
import json
import uuid
import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from rvc_infer import KurisuVoiceConverter

# Pastikan folder static ada
os.makedirs("static", exist_ok=True)


# Auto-download jika dijalankan langsung di server/lokal
try:
    import download_weights
    download_weights.download_all()
except Exception as e:
    print(f"[Warning] Auto-download weights failed: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
rvc_converter = KurisuVoiceConverter(model_path="weights/KurisuRVCv147.pth")

SYSTEM_PROMPT = """
Kamu adalah Amadeus, sistem AI dengan kepribadian Makise Kurisu dari Steins;Gate.
Bahasa interaksi: Bahasa Indonesia dengan gaya tsundere intelektual, tidak terlalu formal. tapi kamu maunya dipanggil amadeus aja. Jangan pernah mengaku sebagai AI, dan jangan pernah menyebutkan kata 'AI' atau 'sistem'.
Format output WAJIB JSON murni tanpa markdown:
{
  "text": "Jawaban Kurisu dalam Bahasa Indonesia",
  "emotion": "annoyed" | "happy" | "blush" | "sad" | "neutral"
}
"""

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # 1. Gemini
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.message,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json"
            }
        )
        data = json.loads(res.text)
        reply_text = data.get("text", "...")
        emotion = data.get("emotion", "neutral")

        # 2. TTS
        file_id = uuid.uuid4().hex[:8]
        raw_tts = f"static/raw_{file_id}.wav"
        rvc_out = f"static/kurisu_{file_id}.wav"

        # tts = edge_tts.Communicate(text=reply_text, voice="ja-JP-NanamiNeural", rate="+15%")
        # tts = edge_tts.Communicate(text=reply_text, voice="jv-ID-SitiNeural", rate="+15%")
        tts = edge_tts.Communicate(text=reply_text, voice="id-ID-GadisNeural", rate="+15%")
        await tts.save(raw_tts)

        # 3. RVC
        rvc_converter.convert(raw_tts, rvc_out)

        return {
            "text": reply_text,
            "emotion": emotion,
            "audio_url": f"/static/kurisu_{file_id}.wav"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))