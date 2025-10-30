# import streamlit as st
# from langchain_community.llms import Ollama  # Updated import path

# # Streamlit title
# st.title("Hi, write your prompt!")

# # Input for the prompt
# prompt = st.text_area(label="Write your prompt.")
# button = st.button("Okay")

# if button:
#     if prompt:
#         # Initialize the local LLM
#         llm = Ollama(model='llama3.1')  # Specify your model here

#         # Generate a response using the local LLM
#         response = llm(prompt)

#         # Display the response
#         st.markdown(response)
import streamlit as st
from langchain_community.llms import Ollama  # legacy location (deprecated)

st.title("TTS model")

prompt = st.text_area("Write your prompt.")
if st.button("Okay") and prompt:
    llm = Ollama(model="qwen3:0.6b", base_url="http://localhost:11434")
    response = llm.invoke(prompt)   # or: llm.predict(prompt)
    st.markdown(response)
