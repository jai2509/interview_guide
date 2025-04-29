import streamlit as st
import os
import pdfplumber
import PyPDF2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from huggingface_hub import InferenceClient
from io import BytesIO
import base64
import pyspark
import time
import tempfile

st.set_page_config(page_title="SmartHire AI", layout="centered")

load_dotenv()
nlp = spacy.load("en_core_web_sm")
groq_api_key = os.getenv("GROQ_API_KEY")
client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2")

st.title("SmartHire AI - Interview & Job Assistant")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    st.success("Resume parsed successfully!")

    doc = nlp(text)
    name = ""
    email = ""
    skills = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        elif ent.label_ == "EMAIL" and not email:
            email = ent.text

    tokens = [token.text.lower() for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
    skills = list(set(tokens))

    st.subheader("Extracted Details")
    st.write(f"**Name:** {name}")
    st.write(f"**Email:** {email}")
    st.write(f"**Skills:** {', '.join(skills[:10])}")

    st.subheader("Ask something about your resume or job suggestions")
    user_query = st.text_input("Your Question:")

    if user_query:
        resume_embedding = client.embed(text)
        query_embedding = client.embed(user_query)
        similarity = cosine_similarity([resume_embedding], [query_embedding])[0][0]

        st.write(f"**Similarity Score:** {similarity:.2f}")

        response_text = "Based on your resume and query, here are some insights or suggestions:"
        if similarity > 0.5:
            response_text += " You have a good match for roles like Software Developer, Data Analyst, or AI Engineer."
        else:
            response_text += " You may want to focus more on tailoring your resume to specific roles."

        st.success(response_text)

        st.subheader("Audio Response")
        tts = gTTS(text=response_text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tts.save(tmp_audio.name)
            audio_file = open(tmp_audio.name, "rb")
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            st.audio(f"data:audio/mp3;base64,{audio_b64}", format="audio/mp3")

    st.subheader("Job Recommendations from Jooble")
    jooble_api = os.getenv("JOOBLE_API")
    if jooble_api:
        query = st.text_input("Enter job title to search jobs:")
        if st.button("Search Jobs") and query:
            payload = {
                "keywords": query,
                "location": "India"
            }
            response = requests.post(
                f"https://jooble.org/api/{jooble_api}",
                json=payload
            )
            if response.status_code == 200:
                jobs = response.json().get("jobs", [])
                for job in jobs[:5]:
                    st.markdown(f"**{job['title']}** at {job['company']}")
                    st.write(f"Location: {job['location']}")
                    st.write(f"Link: [Apply Here]({job['link']})")
                    st.write("---")
            else:
                st.error("Failed to fetch jobs from Jooble API")

else:
    st.info("Please upload your resume to get started.")
