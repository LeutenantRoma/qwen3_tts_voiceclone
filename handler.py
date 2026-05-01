import runpod
import base64
import io
import soundfile as sf
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


# Model loading and text processing
# -------------------------------------------------------


def decode_base64_audio(base64_string):
    audio_bytes = base64.b64decode(base64_string)
    buffer = io.BytesIO(audio_bytes)
    waveform, sr = sf.read(buffer)
    return waveform, sr

def synthesize(text: str, ref_audio: str, ref_text: str, language: str):
    
    model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    torch_dtype=torch.float16  # safer than bf16 unless you're on A100
    )
    
    waveform_ref, sr_ref = decode_base64_audio(ref_audio)
    ref_text = ref_text

    audio_list, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=(waveform_ref, sr_ref),
        ref_text=ref_text,
    )

    waveform = audio_list[0]
    waveform = np.asarray(waveform, dtype=np.float32)

    # Flatten if needed
    if waveform.ndim > 1:
        waveform = waveform.squeeze()

    return waveform, sr

# Handler itself
# --------------------------------------------------------------
def handler(job):
    text = job["input"].get("text", "")
    ref_audio = job["input"].get("ref_audio", "")
    ref_text = job["input"].get("ref_text", "")

    waveform, sr = synthesize(text,ref_audio,ref_text)
    waveform = np.asarray(waveform, dtype="float32")

    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="WAV")
    buffer.seek(0)

    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "audio": audio_base64
    }

# Star the job
# ---------------------------------------------------
runpod.serverless.start({"handler": handler})
