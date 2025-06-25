from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import whisper
from util import load_audio

app = FastAPI()
voice_to_text_model = whisper.load_model("small", device="cpu")  # "tiny", "base", "small", "medium" models are free

@app.get("/")
async def root():
    return {"message": "Hello World"}


def voice_to_text_audio_file(uploaded_file):
    result = voice_to_text_model.transcribe(uploaded_file, fp16=False)
    return result["text"]


@app.post("/voice")
async def upload_voice(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio_numpy = load_audio(audio_bytes)

        file_content = voice_to_text_audio_file(audio_numpy)

        return {"message": file_content}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )
