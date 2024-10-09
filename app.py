import streamlit as st
from transformers import pipeline

# Set up the text generation pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")

# Streamlit app
st.title("FLAN-T5 Chatbot")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input.lower() != "bye":
        # Generate the response using the pipeline
        bot_response = pipe(user_input)[0]['generated_text']
        st.text_area("Chatbot:", value=bot_response, height=150, disabled=True)
    else:
        st.text_area("Chatbot:", value="Goodbye!", height=150, disabled=True)
