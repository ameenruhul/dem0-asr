# app.py
import streamlit as st
import torch
import numpy as np
import io
import warnings
import soundfile as sf
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from peft import PeftModelForSeq2SeqLM

@st.cache_resource(show_spinner=False)
def load_asr():
    """Load Whisper v3 + LoRA adapter once."""
    base = WhisperForConditionalGeneration.from_pretrained(
        "unsloth/whisper-large-v3",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModelForSeq2SeqLM.from_pretrained(
        base,
        "./final_optimized_model",
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    proc = WhisperProcessor.from_pretrained("unsloth/whisper-large-v3")
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        return_language=True,
        torch_dtype=torch.float16
    )
    # Force Bengali
    asr.model.generation_config.language = "bn"
    asr.model.generation_config.forced_decoder_ids = None
    return asr

def transcribe(asr, audio_bytes):
    """Read audio bytes via soundfile and run inference."""
    # Read into NumPy array
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    # If stereo, convert to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Normalize to [-1,1]
    if data.dtype.kind == "i":
        data = data / np.iinfo(data.dtype).max
    # Run pipeline (long audio auto‐chunks under the hood)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        out = asr({"array": data, "sampling_rate": sr}, return_timestamps=False)
    return out["text"]

def main():
    st.set_page_config(page_title="Bengali Whisper ASR", layout="wide")
    st.title("🎙️ Bengali Whisper ASR Demo")
    st.write("Upload a WAV/FLAC file (librosa/libsndfile‐supported) to transcribe in Bengali.")

    uploaded = st.file_uploader("Choose audio", type=["wav","flac"])
    if uploaded:
        st.audio(uploaded, format=uploaded.type)
        asr = load_asr()
        with st.spinner("Transcribing…"):
            text = transcribe(asr, uploaded.getvalue())
        st.subheader("Transcription")
        st.write(text)

if __name__ == "__main__":
    main()
