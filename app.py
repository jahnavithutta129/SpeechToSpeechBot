
import streamlit as st
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.eos_token_id)

def generate_text(prompt):
    numeric_ids = tokenizer.encode(prompt, return_tensors='pt')
    result = model.generate(numeric_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
    return generated_text

def recognize_speech_from_file(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Streamlit app layout
st.title("Speech-to-Text and Text Generation App")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    recognized_text = recognize_speech_from_file("temp_audio.wav")
    st.write(f"Recognized Text: {recognized_text}")

    if recognized_text:
        generated_text = generate_text(recognized_text)
        st.write("Generated Text:")
        st.write(generated_text)

    os.remove("temp_audio.wav")
