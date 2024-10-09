import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained FLAN-T5 large model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate chatbot responses
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    output = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Streamlit app
st.title("FLAN-T5 Chatbot")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input.lower() != "bye":
        bot_response = generate_response(user_input)
        st.text_area("Chatbot:", value=bot_response, height=150, disabled=True)
    else:
        st.text_area("Chatbot:", value="Goodbye!", height=150, disabled=True)
