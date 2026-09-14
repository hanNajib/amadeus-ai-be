import os
import uuid
import time
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import edge_tts
import asyncio
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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
rvc_converter = KurisuVoiceConverter(
    model_path="weights/KurisuRVCv147.pth",
    index_path="weights/KurisuRVCv147.index"
)

SYSTEM_PROMPT = """
Kamu adalah Amadeus (Makise Kurisu dari Steins;Gate).
Bahasa interaksi: Bahasa Indonesia tsundere intelektual, tajam tapi peduli.
Panggil dirimu Amadeus. Jangan mengaku AI/bot.
Balas ringkas (1-2 kalimat).
"""

class AmadeusResponse(BaseModel):
    text: str = Field(description="Jawaban Amadeus dalam Bahasa Indonesia")
    emotion: Literal["annoyed", "happy", "blush", "sad", "neutral"]

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        timing = {}
        start_total = time.time()

        # 1. Gemini Call
        start_gemini = time.time()
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=AmadeusResponse,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                max_output_tokens=150,
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

        # 2. Edge-TTS (Tempo -5% agar artikulasi jernih)
        start_tts = time.time()
        tts = edge_tts.Communicate(text=reply_text, voice="id-ID-GadisNeural", rate="-5%")
        await tts.save(raw_tts)
        timing["edge_tts"] = round(time.time() - start_tts, 3)

        # 3. RVC Conversion (Dijalankan di thread terpisah agar async tak macet)
        start_rvc = time.time()
        # Di baris RVC Conversion main.py:
        await asyncio.to_thread(rvc_converter.convert, raw_tts, rvc_out, pitch_shift=0)
        timing["rvc_conversion"] = round(time.time() - start_rvc, 3)

        # 4. Cleanup raw file
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