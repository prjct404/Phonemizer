# # from assets.GE2PE import GE2PE
# from GE2PE import GE2PE
# import os
# os.environ["PYTHONUTF8"] = "1"

# g2p = GE2PE(model_path='model-weights/homo-ge2pe') # or homo-t5

# x=g2p.generate(['تست مدل تبدیل نویسه به واج', 'این کتابِ علی است'], use_rules=True)
# print(x)


import os
import streamlit as st
from GE2PE import GE2PE

os.environ["PYTHONUTF8"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TensorFlow info logs

st.title("🗣️ GE2PE – Persian Grapheme to Phoneme")

text = st.text_area("Write Persian text:")
if st.button("Convert") and text:
    g2p = GE2PE(model_path="model-weights/homo-ge2pe")  # or "homo-t5"
    result = g2p.generate([text], use_rules=True)
    st.markdown(f"**Phoneme Output:** {result[0]}")
