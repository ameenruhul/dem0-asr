# app.py

import streamlit as st
import io
import os
import numpy as np
import torch
import soundfile as sf
import warnings

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from peft import PeftModelForSeq2SeqLM

# change these to match your repo layout:
BASE_DIR = os.path.dirname(__file__)
BASE_MODEL_DIR = os.path.join(BASE_DIR, "whisper-large-v3")
PEFT_MODEL_DIR = os.path.join(BASE_DIR, "final_optimized_model")

@st.cache_resource(show_spinner=False)
def load_asr():
    """Load base Whisper and LoRA adapter into an ASR pipeline, all offline."""
    # 1) Load the base multilingual Whisper from local folder
    if not os.path.isdir(BASE_MODEL_DIR):
        raise FileNotFoundError(f"Base Whisper model folder not found at {BASE_MODEL_DIR}")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    # 2) Load your fine-tuned LoRA weights from the local folder
    if not os.path.isdir(PEFT_MODEL_DIR):
        raise FileNotFoundError(f"LoRA weights folder not found at {PEFT_MODEL_DIR}")
    model = PeftModelForSeq2SeqLM.from_pretrained(
        base,
        PEFT_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    ).eval()

    # 3) Load processor (feature_extractor + tokenizer) from same local folder
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
    )

    # 4) Build the HF ASR pipeline
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_language=True,
        torch_dtype=torch.float16,
    )
    return asr

def transcribe(asr, audio_bytes: bytes) -> str:
    """Read raw bytes via soundfile, send to the pipeline, return text."""
    audio_buffer = io.BytesIO(audio_bytes)
    data, sr = sf.read(audio_buffer)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = asr({"array": data, "sampling_rate": sr})

    return out["text"]

def main():
    st.set_page_config(page_title="Whisper-LoRA ASR Demo", layout="wide")
    st.title("🎙️ Bengali Whisper-LoRA Transcription")
    st.caption("App version: v1.0.1")

    st.markdown(
        """
        Upload a Bengali audio file (wav, flac, m4a, mp3…).  
        Everything runs offline from your local model folders.
        """
    )

    uploaded = st.file_uploader(
        "Choose an audio file",
        type=["wav", "flac", "mp3", "m4a", "aac"]
    )
    if not uploaded:
        st.info("Waiting for you to upload an audio file.")
        return

    st.audio(uploaded, format=uploaded.type)

    asr = load_asr()
    with st.spinner("Transcribing… this may take a few seconds for long files"):
        text = transcribe(asr, uploaded.getvalue())

    st.subheader("Transcription")
    st.write(text)

if __name__ == "__main__":
    main()
