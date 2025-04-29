import streamlit as st
import pdfplumber
import spacy
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import requests
import os
import tempfile
import shutil
from email.message import EmailMessage

# Load environment variables
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to perform basic NLP on text
def analyze_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return tokens, entities

# Function to capture audio and transcribe
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Sorry, service unavailable."

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_path = fp.name
        tts.save(temp_path)
    audio = AudioSegment.from_mp3(temp_path)
    play(audio)
    os.remove(temp_path)

# Streamlit App
def main():
    st.title("🧠 AI Interview Guide")

    menu = ["Home", "Resume Analysis", "Speech Recognition", "Text-to-Speech"]
    choice = st.sidebar.selectbox("Select Activity", menu)

    if choice == "Home":
        st.subheader("Welcome to the AI Interview Guide!")
        st.write("Upload your resume, analyze it, practice speaking, and receive voice feedback!")

    elif choice == "Resume Analysis":
        st.subheader("📄 Analyze Your Resume")
        pdf_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
        if pdf_file is not None:
            text = extract_text_from_pdf(pdf_file)
            st.text_area("Extracted Text", text, height=300)

            tokens, entities = analyze_text(text)

            st.subheader("Tokens:")
            st.write(tokens)

            st.subheader("Named Entities:")
            st.write(entities)

            # Simple entity visualization
            if entities:
                labels = [label for (_, label) in entities]
                label_freq = {label: labels.count(label) for label in set(labels)}
                plt.bar(label_freq.keys(), label_freq.values())
                plt.title("Named Entity Distribution")
                plt.xlabel("Entity Type")
                plt.ylabel("Count")
                st.pyplot(plt)

    elif choice == "Speech Recognition":
        st.subheader("🎤 Speech Recognition")
        if st.button("Record Speech"):
            result = recognize_speech()
            st.success(f"You said: {result}")

    elif choice == "Text-to-Speech":
        st.subheader("🗣️ Text-to-Speech")
        user_text = st.text_area("Enter text to speak:")
        if st.button("Speak"):
            if user_text:
                text_to_speech(user_text)
            else:
                st.warning("Please enter some text first.")

if __name__ == '__main__':
    main()
