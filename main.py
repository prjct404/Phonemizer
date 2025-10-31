# app_min_edge_tts_two_speech.py
# Requires:
#   pip install streamlit openai edge-tts
#   Set env key: OPENROUTER_API_KEY=sk-or-...

import os
import io
import asyncio
import streamlit as st
from openai import OpenAI
from GE2PE import GE2PE
import edge_tts

# --- constants ---
MODEL_PATH = "model-weights/homo-t5"
PROMPT_FILE = "prompt_base.txt"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
VOICE = "fa-IR-DilaraNeural"  # female voice (or "fa-IR-FaridNeural" male)
os.environ["OPENROUTER_API_KEY"]="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"

REPLACEMENTS = {"a":"A", "$":"S", "/":"a", "1":"", ";":"Z", "@":"?", "c":"C"}

# --- helpers ---
def read_prompt(path):
    try:
        return open(path, "r", encoding="utf-8").read().strip()
    except:
        return "You are a helpful assistant."

@st.cache_resource
def load_g2p(path):
    return GE2PE(model_path=path)

def replace_chars(s):
    return "".join(REPLACEMENTS.get(ch, ch) for ch in s)

def init_client():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        st.error("❗ Please set OPENROUTER_API_KEY environment variable.")
        st.stop()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

def call_llm(client, model_id, sys, user):
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content

async def _speak_async(text, voice=VOICE):
    communicate = edge_tts.Communicate(text, voice=voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()

def speak(text):
    return asyncio.run(_speak_async(text))

# --- app UI ---
st.title("🗣️ GE2PE → LLM → TWO Persian Speech Outputs")

base_prompt = read_prompt(PROMPT_FILE)
g2p = load_g2p(MODEL_PATH)
client = init_client()

text = st.text_area("✍️ Enter Persian text:", height=150)

if st.button("Convert + Send to LLM + Speak Both"):
    if not text.strip():
        st.error("Please enter text.")
    else:
        # GE2PE
        raw = g2p.generate([text], use_rules=True)
        phoneme = replace_chars(raw[0])
        st.subheader("finglish Phoneme Output")
        st.code(phoneme)

        # LLM
        st.subheader("🤖 LLM Output")
        ai_text = call_llm(client, OPENROUTER_MODEL, base_prompt, phoneme)
        st.write(ai_text)

        # TTS #1 — Original input
        st.subheader("🔊 Speech: Original Input")
        audio1 = speak(text)
        st.audio(audio1, format="audio/mp3")
        st.download_button("Download Input Speech", audio1, "input_fa.mp3")

        # TTS #2 — LLM Output (must be Persian text)
        st.subheader("🔊 Speech: LLM Output")
        audio2 = speak(ai_text)
        st.audio(audio2, format="audio/mp3")
        st.download_button("Download LLM Speech", audio2, "llm_fa.mp3")
