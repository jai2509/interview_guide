import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import smtplib
import ssl
from email.message import EmailMessage
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
import json
import zipfile
import tempfile

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Constants
DRIVE_FILE_ID = "1RxxUscA5xQqLRJXLgrRQrCrTphKN9QyQ"
GDRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# Cache data load
@st.cache_data
def download_and_extract_dataset():
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    try:
        response = requests.get(GDRIVE_URL)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extract_path = tempfile.mkdtemp()
            zip_ref.extractall(extract_path)
        csv_files = [os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.endswith(".csv")]
        if csv_files:
            return pd.read_csv(csv_files[0])
    except Exception as e:
        st.error(f"Error downloading dataset: {e}")
        return pd.DataFrame()

dataset = download_and_extract_dataset()

# Resume parsing
def parse_resume(pdf_file):
    reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Get questions from dataset
def get_questions_from_dataset(role):
    if dataset.empty or 'role' not in dataset.columns:
        return []
    questions = dataset[dataset['role'].str.lower() == role.lower()]
    return questions['questions'].tolist()[:5] if not questions.empty else []

# Score answers using GROQ
def score_answers(questions, answers):
    if not questions or not answers:
        return 0, []
    prompt = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(questions, answers))])
    messages = [{"role": "user", "content": f"Evaluate these answers and provide a score out of 10:\n{prompt}"}]
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"model": "mixtral-8x7b-32768", "messages": messages}
    )
    result = response.json()
    try:
        feedback = result["choices"][0]["message"]["content"]
        score = int([int(s) for s in feedback.split() if s.isdigit()][0])
    except:
        score = 0
        feedback = "Could not extract score."
    return score, feedback

# Job recommendations
def get_job_recommendations(title, location="India"):
    url = f"https://jooble.org/api/{JOOBLE_API_KEY}"
    payload = {"keywords": title, "location": location}
    try:
        response = requests.post(url, json=payload)
        jobs = response.json().get('jobs', [])[:3]
        return [(job["title"], job["location"], job["link"]) for job in jobs]
    except:
        return []

# Send email
def send_email(to, subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["From"] = EMAIL_USER
    msg["To"] = to
    msg["Subject"] = subject
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# App UI
st.set_page_config(page_title="SmartHire AI", layout="centered")
st.title("🤖 SmartHire AI Interview Assistant")

menu = st.sidebar.selectbox("Menu", ["Candidate", "Admin Dashboard"])

if menu == "Candidate":
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    role = st.selectbox("Choose Role", dataset["role"].unique() if not dataset.empty else ["Data Scientist", "ML Engineer"])
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

    if resume_file:
        resume_text = parse_resume(resume_file)
        st.success("Resume parsed successfully.")

    if st.button("Start Interview"):
        if not resume_file or not role:
            st.warning("Please upload a resume and choose a role.")
        else:
            questions = get_questions_from_dataset(role)
            answers = []
            for q in questions:
                answers.append(st.text_input(q, key=q))
            if st.button("Submit Answers"):
                score, feedback = score_answers(questions, answers)
                st.markdown(f"### Score: {score}/10")
                st.markdown("#### Feedback")
                st.write(feedback)

                jobs = get_job_recommendations(role)
                st.markdown("#### Recommended Jobs")
                for title, loc, link in jobs:
                    st.write(f"- [{title} ({loc})]({link})")

                body = f"Hi {name},\n\nYour interview score: {score}/10.\nFeedback:\n{feedback}\n\nGood luck!\nSmartHire AI"
                try:
                    send_email(email, "Your SmartHire AI Interview Results", body)
                    st.success("Results sent via email.")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

                # Save report for admin
                report = {
                    "name": name,
                    "email": email,
                    "role": role,
                    "score": score,
                    "feedback": feedback
                }
                if "reports" not in st.session_state:
                    st.session_state["reports"] = []
                st.session_state["reports"].append(report)

elif menu == "Admin Dashboard":
    admin_email = st.text_input("Admin Email")
    admin_pass = st.text_input("Password", type="password")

    if st.button("Login"):
        if admin_email == ADMIN_EMAIL and admin_pass == ADMIN_PASSWORD:
            st.success("Logged in as admin.")
            reports = st.session_state.get("reports", [])
            if reports:
                df = pd.DataFrame(reports)
                st.dataframe(df)
                st.download_button("Download Report CSV", df.to_csv(index=False), "interview_reports.csv")
            else:
                st.warning("No reports found.")
        else:
            st.error("Invalid credentials.")
