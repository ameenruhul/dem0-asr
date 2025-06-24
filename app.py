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

# Path configuration for Streamlit Cloud
# BASE_DIR is the root of your deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_DIR = os.path.join(BASE_DIR, "whisper-large-v3")
PEFT_MODEL_DIR = os.path.join(BASE_DIR, "final_optimized_model")

# Debug info
st.sidebar.markdown("### Debug Info")
st.sidebar.write(f"Base directory: {BASE_DIR}")
st.sidebar.write(f"Base model directory: {BASE_MODEL_DIR}")
st.sidebar.write(f"PEFT model directory: {PEFT_MODEL_DIR}")
st.sidebar.write(f"Base model exists: {os.path.exists(BASE_MODEL_DIR)}")
st.sidebar.write(f"PEFT model exists: {os.path.exists(PEFT_MODEL_DIR)}")

@st.cache_resource(show_spinner=False)
def load_asr():
    """Load base Whisper and LoRA adapter into an ASR pipeline."""
    with st.spinner("Loading ASR model (this may take a minute)..."):
        # Load the processor directly from HF to ensure we have all needed files
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        
        # Load base model
        if not os.path.exists(BASE_MODEL_DIR):
            st.error(f"Base model directory not found at {BASE_MODEL_DIR}")
            st.error(f"Contents of current directory: {os.listdir(BASE_DIR)}")
            raise FileNotFoundError(f"Base model directory not found at {BASE_MODEL_DIR}")
        
        st.sidebar.write(f"Base model directory contents: {os.listdir(BASE_MODEL_DIR)}")
        
        base = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        
        # Load LoRA adapter
        if not os.path.exists(PEFT_MODEL_DIR):
            st.error(f"PEFT model directory not found at {PEFT_MODEL_DIR}")
            st.error(f"Contents of current directory: {os.listdir(BASE_DIR)}")
            raise FileNotFoundError(f"PEFT model directory not found at {PEFT_MODEL_DIR}")
        
        st.sidebar.write(f"PEFT model directory contents: {os.listdir(PEFT_MODEL_DIR)}")
        
        model = PeftModelForSeq2SeqLM.from_pretrained(
            base, 
            PEFT_MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        ).eval()
        
        # Create ASR pipeline
        asr = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps=False,
            torch_dtype=torch.float16,
        )
        
        return asr

def transcribe(asr, audio_bytes: bytes) -> str:
    """Read raw bytes via soundfile, send to the pipeline, return text."""
    audio_buffer = io.BytesIO(audio_bytes)
    
    try:
        data, sr = sf.read(audio_buffer)
    except Exception as e:
        st.error(f"Error reading audio file: {e}")
        return "Error processing audio file. Please try another file format."
    
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = asr({"array": data, "sampling_rate": sr})
        return result["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return "Error during transcription. Please try again with a different audio file."

def main():
    st.set_page_config(page_title="Bengali Whisper-LoRA ASR Demo", layout="wide")
    st.title("🎙️ Bengali Whisper-LoRA Transcription")
    st.caption("App version: v1.0.2")
    
    st.markdown(
        """
        Upload a Bengali audio file (wav, flac, m4a, mp3...) to get a transcription.
        This app uses a fine-tuned Whisper model specifically for Bengali speech recognition.
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
    
    if st.button("Transcribe Audio"):
        try:
            asr = load_asr()
            
            with st.spinner("Transcribing... this may take a few seconds for long files"):
                text = transcribe(asr, uploaded.getvalue())
            
            st.subheader("Transcription")
            st.write(text)
        except Exception as e:
            st.error(f"Error loading model or transcribing: {str(e)}")
            st.info("Please check the debug information in the sidebar for more details.")

if __name__ == "__main__":
    main()