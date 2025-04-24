# app.py
import streamlit as st
import pdfplumber
import os
import spacy
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
import tempfile
import re
import requests
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# 📄 Resume Parsing
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return " ".join(page.extract_text() or '' for page in pdf.pages)

def extract_skills(text):
    doc = nlp(text)
    return list(set(ent.text.lower() for ent in doc.ents if ent.label_ in ['ORG', 'SKILL', 'PRODUCT', 'WORK_OF_ART']))

# ---------------------------
# 🎤 Voice Input
# ---------------------------
def record_audio(language="en-US"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language=language)
    except:
        st.error("Sorry, I could not understand your voice.")
        return ""

# ---------------------------
# 🔊 Voice Output
# ---------------------------
def play_text(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = tempfile.mktemp(suffix=".mp3")
    tts.save(filename)
    audio = AudioSegment.from_file(filename)
    play(audio)

# ---------------------------
# 🤖 Groq LLaMA3 Integration
# ---------------------------
def call_groq_llama3(prompt, model="llama3-8b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def generate_questions(role, skills, language_name):
    prompt = f"""You're a technical interviewer. The candidate is applying for the role of {role}.
    They have skills: {', '.join(skills)}.
    Ask 5 technical and 2 behavioral interview questions.
    Respond only in {language_name}.
    Format: Q1: ... Q2: ...
    """
    response = call_groq_llama3(prompt)
    return re.findall(r"Q\d+: (.+)", response)

def generate_feedback(role, resume_summary, qas, language_name):
    qas_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in qas])
    prompt = f"""You are an AI Interview Evaluator. Candidate applied for {role}.
    Resume Summary: {resume_summary}

    Interview Transcript:
    {qas_str}

    Provide in {language_name}:
    - Strengths
    - Weaknesses
    - Technical Score (1-10)
    - Communication Score (1-10)
    - Fit for the Role
    - Final Verdict
    - Suggested Learning Resources
    """
    return call_groq_llama3(prompt)

def extract_scores(feedback):
    tech = re.search(r"Technical Score.*?(\d+)", feedback)
    comm = re.search(r"Communication Score.*?(\d+)", feedback)
    return int(tech.group(1)) if tech else 0, int(comm.group(1)) if comm else 0

def display_dashboard(tech_score, comm_score):
    st.subheader("📊 Interview Metrics")
    fig, ax = plt.subplots()
    ax.bar(["Technical", "Communication"], [tech_score, comm_score], color=['steelblue', 'orange'])
    ax.set_ylim([0, 10])
    st.pyplot(fig)

# ---------------------------
# 📧 Email Feedback
# ---------------------------
def send_feedback_email(recipient_email, feedback_text):
    msg = EmailMessage()
    msg['Subject'] = "Your AI Interview Feedback"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient_email
    msg.set_content(feedback_text)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------------
# 🎯 Streamlit App
# ---------------------------
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("🤖 AI Interviewer with Resume, Role-Based Q&A & Multilingual Voice")

language_map_stt = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "Spanish": "es-ES",
    "French": "fr-FR"
}

tts_lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr"
}

stt_lang_choice = st.selectbox("🌐 Choose Input Language", list(language_map_stt.keys()))
tts_lang_choice = st.selectbox("🎧 Choose Output Voice Language", list(tts_lang_map.keys()))

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    skills = extract_skills(text)
    st.success("✅ Resume parsed.")
    st.write("**Extracted Skills:**", skills)

    roles = ["Data Scientist", "Software Engineer", "Product Manager", "AI/ML Engineer"]
    role = st.selectbox("💼 Choose a Job Role for Interview", roles)

    if st.button("Start Interview"):
        questions = generate_questions(role, skills, stt_lang_choice)
        st.session_state.qas = []
        st.session_state.questions = questions
        st.session_state.step = 0

if "questions" in st.session_state and st.session_state.step < len(st.session_state.questions):
    question = st.session_state.questions[st.session_state.step]
    st.subheader(f"Q{st.session_state.step + 1}: {question}")
    if st.button("🔊 Hear Question"):
        play_text(question, lang=tts_lang_map[tts_lang_choice])

    if st.button("🎤 Answer via Voice"):
        answer = record_audio(language=language_map_stt[stt_lang_choice])
        st.text_area("📝 Your Answer (Edit if needed)", value=answer, key="answer")

    if st.button("Next"):
        q = st.session_state.questions[st.session_state.step]
        a = st.session_state.get("answer", "")
        st.session_state.qas.append((q, a))
        st.session_state.step += 1

elif "qas" in st.session_state and st.session_state.step == len(st.session_state.questions):
    st.success("✅ Interview Complete. Generating Feedback...")
    resume_summary = " ".join(skills)
    feedback = generate_feedback(role, resume_summary, st.session_state.qas, stt_lang_choice)
    st.session_state.feedback = feedback
    st.text_area("📄 Personalized Feedback", feedback, height=300)

    if st.button("🔊 Read Feedback"):
        play_text(feedback, lang=tts_lang_map[tts_lang_choice])

    tech_score, comm_score = extract_scores(feedback)
    display_dashboard(tech_score, comm_score)

    user_email = st.text_input("\ud83d\udce7 Enter your email to receive this report")
    if st.button("\ud83d\udce4 Send Feedback to Email"):
        if send_feedback_email(user_email, feedback):
            st.success("\u2705 Feedback sent successfully!")
