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
from peft import PeftConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors

@st.cache_resource(show_spinner=False)
def load_asr():
    """Load base Whisper + manually attach your local LoRA adapter."""
    # 1) Base multilingual Whisper in float16, auto‐sharded on GPU/CPU
    base = WhisperForConditionalGeneration.from_pretrained(
        "unsloth/whisper-large-v3",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2) Load your adapter config & build the PEFT wrapper
    adapter_dir = "/workspace/training_outputs/final_optimized_model"
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")

    if not os.path.isfile(cfg_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError(
            "Couldn’t find adapter_config.json or adapter_model.safetensors in:\n"
            f"  {adapter_dir}\n"
            "Did you run `peft_model.save_pretrained(...)` when training?"
        )

    peft_config = PeftConfig.from_json_file(cfg_path)
    model = get_peft_model(base, peft_config)

    # 3) Load the LoRA weights from your .safetensors
    state_dict = load_safetensors(weights_path, framework="pt")
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # 4) Processor stays the same
    processor = WhisperProcessor.from_pretrained("unsloth/whisper-large-v3")

    # 5) Build the HF ASR pipeline
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_language=True,
        torch_dtype=torch.float16,
    )

def transcribe(asr, audio_bytes: bytes) -> str:
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf)
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

    st.audio(uploaded, format=uploaded.type)

    try:
        asr = load_asr()
    except Exception as e:
        st.error(str(e))
        return

    with st.spinner("Transcribing… this may take a few seconds for long files"):
        text = transcribe(asr, uploaded.getvalue())

    st.subheader("Transcription")
    st.write(text)

if __name__ == "__main__":
    main()
