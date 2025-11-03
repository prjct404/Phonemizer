import os
import io
import asyncio
import streamlit as st
from openai import OpenAI
from GE2PE import GE2PE
import edge_tts
from edge_tts.exceptions import NoAudioReceived
from Monitoring_metrics import INFER_REQUESTS, INFER_LATENCY

# --- constants ---
MODEL_PATH = "model-weights/homo-t5"
PROMPT_FILE = "prompt_base.txt"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_VOICE = "fa-IR-DilaraNeural"  # default voice (female voice)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
REPLACEMENTS = {"a": "A", "$": "S", "/": "a", "1": "", ";": "Z", "@": "?", "c": "C"}


# --- Persian language validation ---
def is_persian_text(text):
    """
    Check if text is in Persian language.
    Returns (is_persian, non_persian_chars)
    """
    if not text or not text.strip():
        return False, []

    # Persian/Arabic Unicode ranges
    persian_ranges = [
        (0x0600, 0x06FF),  # Arabic block (includes Persian)
        (0x06F0, 0x06F9),  # Persian numbers
        (0x200C, 0x200D),  # Zero-width non-joiner and joiner
        (0x064B, 0x065F),  # Arabic diacritics
    ]

    # Allowed characters: whitespace, punctuation, Persian/Arabic
    allowed_punctuation = set(".,;:!?()[]{}\"'«»،؛")
    allowed_whitespace = set(" \t\n\r")

    non_persian_chars = []
    persian_char_count = 0
    total_char_count = 0

    for char in text:
        if char in allowed_whitespace:
            continue  # Skip whitespace

        total_char_count += 1

        # Check if character is in Persian ranges
        is_persian_char = False
        char_code = ord(char)

        for start, end in persian_ranges:
            if start <= char_code <= end:
                is_persian_char = True
                persian_char_count += 1
                break

        # Allow punctuation
        if char in allowed_punctuation:
            continue

        # If not Persian and not allowed punctuation, it's non-Persian
        if not is_persian_char and char not in allowed_punctuation:
            if char not in non_persian_chars:
                non_persian_chars.append(char)

    if total_char_count == 0:
        return False, []

    persian_ratio = (
        persian_char_count / total_char_count if total_char_count > 0 else 0
    )
    is_persian = (
        persian_char_count > 0
        and (persian_ratio >= 0.7 or len(non_persian_chars) == 0)
    )

    return is_persian, non_persian_chars


def validate_persian_input(text):
    """
    Validate that input text is in Persian.
    Raises ValueError with descriptive message if not Persian.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    is_persian, non_persian_chars = is_persian_text(text)

    if not is_persian:
        non_persian_sample = "".join(non_persian_chars[:10])
        if len(non_persian_chars) > 10:
            non_persian_sample += "..."

        error_msg = (
            f"⚠️ **ERROR: Non-Persian text detected!**\n\n"
            f"This application only accepts Persian (Farsi) text. "
            f"Please enter your text in Persian language only.\n\n"
            f"**Detected non-Persian characters:** `{non_persian_sample}`\n\n"
            f"Please rewrite your input in Persian and try again."
        )
        raise ValueError(error_msg)

    return True


# --- voice selection helpers ---
async def list_voices():
    voices = await edge_tts.list_voices()
    return voices


@st.cache_data(ttl=3600)
def get_available_voices():
    return asyncio.run(list_voices())


# --- helpers ---
def read_prompt(path):
    try:
        return open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        return "You are a helpful assistant."


@st.cache_resource
def load_g2p(path):
    return GE2PE(model_path=path)


def replace_chars(s):
    return "".join(REPLACEMENTS.get(ch, ch) for ch in s)


def init_client():
    key = OPENROUTER_API_KEY
    if not key:
        st.error("❗ Please set OPENROUTER_API_KEY in your .env file.")
        st.stop()
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def call_llm(client, model_id, sys, user):
    resp = client.chat.completions.create(
        model=model_id,
        temperature=0,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


async def _speak_async(text, voice=None):
    voice = voice or DEFAULT_VOICE
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    if not voice:
        raise ValueError("Voice cannot be empty")
    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        if buf.tell() == 0:
            raise ValueError(
                f"No audio received for voice '{voice}'. Please verify the voice name is correct."
            )
        return buf.getvalue()
    except NoAudioReceived as e:
        raise ValueError(
            f"No audio received from edge-tts. Voice: '{voice}'. This may be due to an invalid voice, network issues, or service unavailability. Please try again or select a different voice."
        ) from e


def speak(text, voice=None):
    return asyncio.run(_speak_async(text, voice=voice))


# --- app UI ---
st.title("🗣️ GE2PE → LLM → TWO Persian Speech Outputs")

base_prompt = read_prompt(PROMPT_FILE)
g2p = load_g2p(MODEL_PATH)
client = init_client()

# Voice selection
st.subheader("🎤 Voice Selection")
voices = get_available_voices()

persian_voices = [
    v
    for v in voices
    if (
        v.get("Locale", "").startswith("fa-IR")
        or v.get("ShortName", "").startswith("fa-IR")
    )
]

if persian_voices:
    voice_options = {
        f"{v['ShortName']} ({v['Gender']})": v["ShortName"] for v in persian_voices
    }
    voice_list = list(voice_options.values())
    default_index = (
        voice_list.index(DEFAULT_VOICE) if DEFAULT_VOICE in voice_list else 0
    )
    selected_voice_name = st.selectbox(
        "Choose a voice:",
        options=list(voice_options.keys()),
        index=default_index,
        help="Select a voice for text-to-speech synthesis",
    )
    selected_voice = voice_options[selected_voice_name]
else:
    voice_options = {
        f"{v.get('ShortName', 'Unknown')} ({v.get('Gender', 'Unknown')})": v.get(
            "ShortName", ""
        )
        for v in voices
        if v.get("ShortName")
    }
    voice_list = list(voice_options.values())
    default_index = (
        voice_list.index(DEFAULT_VOICE) if DEFAULT_VOICE in voice_list else 0
    )
    selected_voice_name = st.selectbox(
        "Choose a voice:",
        options=list(voice_options.keys()),
        index=default_index,
        help="Select a voice for text-to-speech synthesis",
    )
    selected_voice = voice_options[selected_voice_name]

valid_voice_names = {v.get("ShortName") for v in voices if v.get("ShortName")}
if selected_voice and selected_voice not in valid_voice_names:
    st.warning(
        f"⚠️ Warning: Selected voice '{selected_voice}' may not be valid. Falling back to default voice '{DEFAULT_VOICE}'."
    )
    if DEFAULT_VOICE in valid_voice_names:
        selected_voice = DEFAULT_VOICE
    elif valid_voice_names:
        selected_voice = list(valid_voice_names)[0]
        st.info(f"Using voice: {selected_voice}")

if selected_voice:
    st.caption(f"Selected voice: `{selected_voice}`")
else:
    st.error("❌ No valid voice selected. Please refresh the page.")
    st.stop()

text = st.text_area(
    "✍️ Enter Persian text:",
    height=150,
    help="⚠️ Only Persian (Farsi) text is accepted. Text in other languages will be rejected.",
    placeholder="ADD Your Persian Text Here...",
)

if st.button("Convert + Send to LLM + Speak Both"):
    if not text.strip():
        st.error("Please enter text.")
    else:
        # Validate Persian input
        try:
            validate_persian_input(text)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        #  Prometheus: count + measure latency around the whole pipeline
        INFER_REQUESTS.inc()
        with INFER_LATENCY.time():
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
            try:
                audio1 = speak(text, voice=selected_voice)
                st.audio(audio1, format="audio/mp3")
                st.download_button(
                    "Download Input Speech", audio1, "input_fa.mp3"
                )
            except Exception as e:
                st.error(f"Error generating speech for input: {str(e)}")
                st.info(
                    f"Voice used: {selected_voice}, Text length: {len(text)}"
                )

            # TTS #2 — LLM Output
            st.subheader("🔊 Speech: LLM Output")
            try:
                audio2 = speak(ai_text, voice=selected_voice)
                st.audio(audio2, format="audio/mp3")
                st.download_button(
                    "Download LLM Speech", audio2, "llm_fa.mp3"
                )
            except Exception as e:
                st.error(f"Error generating speech for LLM output: {str(e)}")
                st.info(
                    f"Voice used: {selected_voice}, Text length: {len(ai_text)}"
                )
