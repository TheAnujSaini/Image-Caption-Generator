import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import torch
import os
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Image Caption Generator", page_icon="🖼️", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    .caption-box {
        padding: 15px;
        background-color: #000;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        font-weight: 500;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #e04343;
    }
    .copy-btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 15px;
        cursor: pointer;
        border-radius: 5px;
        width: 100%;
    }
    .copy-btn:hover {
        background-color: #45a049;
    }
    audio {
        width: 100% !important;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# ---------------- CAPTION GENERATION ----------------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# ---------------- AUDIO PLAYBACK ----------------
def generate_audio(caption_text):
    tts = gTTS(caption_text)
    audio_path = f"caption_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    tts.save(audio_path)
    return audio_path

# ---------------- UI ----------------
st.markdown("<div class='main-title'>🖼️ AI Image Caption Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image & let AI describe it for you</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("✨ Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image)

            st.markdown(f"<div class='caption-box'>{caption}</div>", unsafe_allow_html=True)

            # Copy to clipboard button
            copy_code = f"""
            <button class="copy-btn" onclick="navigator.clipboard.writeText('{caption}')">
                📋 Copy Caption
            </button>
            """
            st.markdown(copy_code, unsafe_allow_html=True)

            # Audio playback
            audio_file = generate_audio(caption)
            audio_bytes = open(audio_file, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
            os.remove(audio_file)
