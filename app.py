import streamlit as st, io, numpy as np, torch, warnings
from pydub import AudioSegment
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration, pipeline
)
from peft import PeftModelForSeq2SeqLM

@st.cache_resource
def load_asr():
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
    asr.model.generation_config.language = "bn"
    asr.model.generation_config.forced_decoder_ids = None
    return asr

def transcribe(asr, audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    samples = np.array(audio.get_array_of_samples(), np.float32)
    if audio.channels > 1:
        samples = samples.reshape(-1,audio.channels).mean(axis=1)
    samples /= np.iinfo(audio.array_type).max
    sr = audio.frame_rate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        out = asr({"array": samples, "sampling_rate": sr})
    return out["text"]

st.title("🎙️ Bengali Whisper ASR")
u = st.file_uploader("Upload audio", type=["wav","mp3","flac","m4a","aac"])
if u:
    st.audio(u)
    asr = load_asr()
    txt = transcribe(asr, u.getvalue())
    st.subheader("Transcription")
    st.write(txt)
