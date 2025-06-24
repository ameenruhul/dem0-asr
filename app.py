# app.py
import streamlit as st
import io, os, requests, zipfile, numpy as np, torch, soundfile as sf, warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from peft import PeftModelForSeq2SeqLM

# ── Paths & Secrets ───────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_DIR   = os.path.join(BASE_DIR, "whisper-large-v3")
PEFT_MODEL_DIR   = os.path.join(BASE_DIR, "final_optimized_model")

# Drive URLs & zip names
BASE_ZIP_URL     = st.secrets["GDRIVE_BASE_ZIP_URL"]
PEFT_ZIP_URL     = st.secrets["GDRIVE_PEFT_ZIP_URL"]
BASE_ZIP_NAME    = st.secrets["BASE_ZIP_NAME"]
PEFT_ZIP_NAME    = st.secrets["PEFT_ZIP_NAME"]

# ── Sidebar Debug ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Debug Info")
st.sidebar.write("Base dir:", BASE_MODEL_DIR, os.path.isdir(BASE_MODEL_DIR))
st.sidebar.write("PEFT dir:", PEFT_MODEL_DIR, os.path.isdir(PEFT_MODEL_DIR))

@st.cache_resource(show_spinner=False)
def load_asr():
    """Ensure both base & PEFT models are on-disk, then load ASR pipeline."""

    # — Download & unzip base model if missing —
    if not os.path.isdir(BASE_MODEL_DIR):
        with st.spinner("Downloading base model…"):
            r = requests.get(BASE_ZIP_URL, stream=True); r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(8192): buf.write(chunk)
        with st.spinner("Unpacking base model…"):
            z = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
            z.extractall(BASE_DIR)
        st.sidebar.success("Base model ready.")

    # — Download & unzip PEFT adapter if missing —
    if not os.path.isdir(PEFT_MODEL_DIR):
        with st.spinner("Downloading PEFT adapter…"):
            r = requests.get(PEFT_ZIP_URL, stream=True); r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(8192): buf.write(chunk)
        with st.spinner("Unpacking PEFT adapter…"):
            z = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
            z.extractall(BASE_DIR)
        st.sidebar.success("PEFT adapter ready.")

    # — Load processor & models —
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR, torch_dtype=torch.float16, device_map="auto", local_files_only=True
    )
    model = PeftModelForSeq2SeqLM.from_pretrained(
        base, PEFT_MODEL_DIR, torch_dtype=torch.float16, device_map="auto", local_files_only=True
    ).eval()

    # — Build pipeline —
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        return_timestamps=False,
        torch_dtype=torch.float16,
    )

def transcribe(asr, audio_bytes: bytes) -> str:
    buf = io.BytesIO(audio_bytes)
    try:
        data, sr = sf.read(buf)
    except Exception as e:
        st.error(f"Audio read error: {e}")
        return ""
    if data.ndim > 1: data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = asr({"array": data, "sampling_rate": sr})
        return res["text"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

def main():
    st.set_page_config("Bengali Whisper-LoRA ASR Demo", layout="wide")
    st.title("🎙️ ASR with Whisper + LoRA")
    st.caption("v1.0.4")
    st.markdown("Upload a Bengali audio file to get a transcription.")

    uploaded = st.file_uploader("Choose audio", type=["wav","flac","mp3","m4a","aac"])
    if not uploaded:
        st.info("Waiting for upload…"); return
    st.audio(uploaded, format=uploaded.type)

    if st.button("Transcribe"):
        asr = load_asr()
        with st.spinner("Transcribing…"):
            txt = transcribe(asr, uploaded.getvalue())
        st.subheader("Transcription")
        st.write(txt)

if __name__ == "__main__":
    main()
