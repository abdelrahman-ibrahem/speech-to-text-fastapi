import soundfile as sf  # Requires `pip install soundfile`
import io
import numpy as np 


def load_audio(file_bytes):
    audio, sr = sf.read(io.BytesIO(file_bytes))
    return audio.astype(np.float32)
