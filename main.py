from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import whisper
import numpy as np
import soundfile as sf
import io
from tempfile import NamedTemporaryFile
import os

app = FastAPI()
voice_to_text_model = whisper.load_model("small", device='cpu')

def load_audio(file_bytes):
    # Load audio and convert to Whisper's required format
    audio, sr = sf.read(io.BytesIO(file_bytes))
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    
    return audio.astype(np.float32)

@app.post("/voice")
async def upload_voice(file: UploadFile = File(...)):
    try:

        audio_bytes = await file.read()
        audio_numpy = load_audio(audio_bytes)
        result = voice_to_text_model.transcribe(audio_numpy, fp16=False)
        
        return {"text": result["text"]}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )