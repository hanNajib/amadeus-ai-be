import os
import uuid
import asyncio
import time
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import edge_tts
from google import genai
from google.genai import types
from rvc_infer import KurisuVoiceConverter

os.makedirs("static", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Inisialisasi client Gemini & model RVC
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
rvc_converter = KurisuVoiceConverter(model_path="weights/KurisuRVCv147.pth")

SYSTEM_PROMPT = """
Kamu adalah Amadeus, dengan kepribadian Makise Kurisu dari Steins;Gate.
Bahasa interaksi: Bahasa Indonesia dengan gaya tsundere intelektual, tajam tapi peduli, tidak terlalu kaku/formal. 
Panggil dirimu Amadeus. Jangan pernah mengaku sebagai AI, bot, atau program komputer.
Balas dengan ringkas (1-2 kalimat) agar respon percakapan tetap natural.
"""

# Schema terstruktur langsung via Pydantic
class AmadeusResponse(BaseModel):
    text: str = Field(description="Jawaban Amadeus Kurisu dalam Bahasa Indonesia")
    emotion: Literal["annoyed", "happy", "blush", "sad", "neutral"]

class ProcessingTime(BaseModel):
    gemini_api: float = Field(description="Waktu pemrosesan Gemini API (detik)")
    edge_tts: float = Field(description="Waktu pemrosesan Edge TTS (detik)")
    rvc_conversion: float = Field(description="Waktu pemrosesan RVC conversion (detik)")
    file_cleanup: float = Field(description="Waktu cleanup file (detik)")
    total: float = Field(description="Total waktu pemrosesan (detik)")

class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    text: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        timing = {}
        start_total = time.time()

        # Gemini API call
        start_gemini = time.time()
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=AmadeusResponse,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                max_output_tokens=200,
                temperature=0.7,
            ),
        )
        timing["gemini_api"] = round(time.time() - start_gemini, 3)

        res_data = AmadeusResponse.model_validate_json(response.text)
        reply_text = res_data.text
        emotion = res_data.emotion

        file_id = uuid.uuid4().hex[:8]
        raw_tts = f"static/raw_{file_id}.wav"
        rvc_out = f"static/kurisu_{file_id}.wav"

        # Edge TTS
        start_tts = time.time()
        tts = edge_tts.Communicate(text=reply_text, voice="id-ID-GadisNeural", rate="+15%")
        await tts.save(raw_tts)
        timing["edge_tts"] = round(time.time() - start_tts, 3)

        # RVC Conversion
        start_rvc = time.time()
        await asyncio.to_thread(rvc_converter.convert, raw_tts, rvc_out)
        timing["rvc_conversion"] = round(time.time() - start_rvc, 3)

        # File Cleanup
        start_cleanup = time.time()
        if os.path.exists(raw_tts):
            os.remove(raw_tts)
        timing["file_cleanup"] = round(time.time() - start_cleanup, 3)

        timing["total"] = round(time.time() - start_total, 3)

        return {
            "text": reply_text,
            "emotion": emotion,
            "audio_url": f"/static/kurisu_{file_id}.wav",
            "processing_time": timing
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def generate_voice(req: TTSRequest):
    try:
        timing = {}
        start_total = time.time()

        file_id = uuid.uuid4().hex[:8]
        raw_tts = f"static/raw_{file_id}.wav"
        rvc_out = f"static/kurisu_{file_id}.wav"

        # Edge TTS
        start_tts = time.time()
        tts = edge_tts.Communicate(text=req.text, voice="id-ID-GadisNeural", rate="+15%")
        await tts.save(raw_tts)
        timing["edge_tts"] = round(time.time() - start_tts, 3)

        # RVC Conversion
        start_rvc = time.time()
        await asyncio.to_thread(rvc_converter.convert, raw_tts, rvc_out)
        timing["rvc_conversion"] = round(time.time() - start_rvc, 3)

        # File Cleanup
        start_cleanup = time.time()
        if os.path.exists(raw_tts):
            os.remove(raw_tts)
        timing["file_cleanup"] = round(time.time() - start_cleanup, 3)

        timing["total"] = round(time.time() - start_total, 3)

        return {
            "audio_url": f"/static/kurisu_{file_id}.wav",
            "processing_time": timing
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice-changer")
async def voice_changer(audio: UploadFile = File(...)):
    try:
        timing = {}
        start_total = time.time()

        file_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(audio.filename)[1]
        if not ext:
            ext = ".wav"
            
        raw_audio = f"static/raw_{file_id}{ext}"
        rvc_out = f"static/kurisu_{file_id}.wav"
        
        # File Upload
        start_upload = time.time()
        with open(raw_audio, "wb") as f:
            f.write(await audio.read())
        timing["file_upload"] = round(time.time() - start_upload, 3)
        
        # RVC Conversion
        start_rvc = time.time()
        await asyncio.to_thread(rvc_converter.convert, raw_audio, rvc_out)
        timing["rvc_conversion"] = round(time.time() - start_rvc, 3)

        # File Cleanup
        start_cleanup = time.time()
        if os.path.exists(raw_audio):
            os.remove(raw_audio)
        timing["file_cleanup"] = round(time.time() - start_cleanup, 3)

        timing["total"] = round(time.time() - start_total, 3)

        return {
            "audio_url": f"/static/kurisu_{file_id}.wav",
            "processing_time": timing
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))