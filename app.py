# app.py

import streamlit as st
import io
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

@st.cache_resource(show_spinner=False)
def load_asr():
    """Load base Whisper and LoRA adapter into an ASR pipeline."""
    # 1) Load the base multilingual Whisper
    base = WhisperForConditionalGeneration.from_pretrained(
        "unsloth/whisper-large-v3",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # 2) Apply your fine-tuned LoRA weights
    model = PeftModelForSeq2SeqLM.from_pretrained(
        base,
        "/workspace/training_outputs/final_optimized_model",
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    # 3) Load processor for both feature extractor & tokenizer
    processor = WhisperProcessor.from_pretrained("unsloth/whisper-large-v3")

    # 4) Build the HF ASR pipeline
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_language=True,        # if you’d like language detection
        torch_dtype=torch.float16,   # match your model dtype
        # no device=… here: device_map="auto" on the model does it
    )
    return asr

def transcribe(asr, audio_bytes: bytes) -> str:
    """Read raw bytes via soundfile, send to the pipeline, return text."""
    # 1) Load into numpy array + sampling rate
    audio_buffer = io.BytesIO(audio_bytes)
    data, sr = sf.read(audio_buffer)  # data: np.ndarray shape (n,) or (n,channels)
    # 2) If stereo/multi, average to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # 3) Normalize if ints
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # 4) Run ASR pipeline (will resample internally via torchaudio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = asr({"array": data, "sampling_rate": sr})

    return out["text"]

def main():
    st.set_page_config(page_title="Whisper-LoRA ASR Demo", layout="wide")
    st.title("🎙️ Bengali Whisper-LoRA Transcription")

    st.markdown(
        """
        Upload a Bengali audio file (wav, flac, m4a, mp3…).  
        The model will auto-chunk long audio and run on GPU if available.
        """
    )

    uploaded = st.file_uploader(
        "Choose an audio file", type=["wav","flac","mp3","m4a","aac"]
    )
    if not uploaded:
        st.info("Waiting for you to upload an audio file.")
        return

    # Play back the upload
    st.audio(uploaded, format=uploaded.type)

    # Load (or grab cached) pipeline
    asr = load_asr()

    # Transcribe
    with st.spinner("Transcribing… this may take a few seconds for long files"):
        text = transcribe(asr, uploaded.getvalue())

    st.subheader("Transcription")
    st.write(text)

if __name__ == "__main__":
    main()
