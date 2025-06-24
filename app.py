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

# Application version
VERSION = "v1.0.0"

@st.cache_resource(show_spinner=False)
def load_asr():
    """Load base Whisper and LoRA adapter into an ASR pipeline."""
    # 1) Load the base multilingual Whisper (cached locally)
    base = WhisperForConditionalGeneration.from_pretrained(
        "unsloth/whisper-large-v3",
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,       # don’t try to download
    )

    # 2) Load your fine-tuned LoRA weights from the local folder
    peft_dir = os.path.join(os.path.dirname(__file__), "final_optimized_model")
    if not os.path.isdir(peft_dir):
        raise FileNotFoundError(f"LoRA weights folder not found at {peft_dir}")
    model = PeftModelForSeq2SeqLM.from_pretrained(
        base,
        peft_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,       # don’t try to download
    ).eval()

    # 3) Load processor for both feature extractor & tokenizer
    processor = WhisperProcessor.from_pretrained(
        "unsloth/whisper-large-v3",
        local_files_only=True
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

    # Display version
    st.caption(f"App version: {VERSION}")

    st.markdown(
        """
        Upload a Bengali audio file (wav, flac, m4a, mp3…).  
        The model will auto-chunk long audio and run on GPU if available.
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
