# SmartHire AI - Streamlit Version (Fully Integrated with Jooble & PySpark)

import streamlit as st
import pandas as pd
import os
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import zipfile
import gdown

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@smarthireai.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
REPORTS_FILE = os.path.join(DATA_FOLDER, "interview_reports.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Helper: Download & unzip LinkedIn dataset
@st.cache_resource
def download_and_extract_dataset():
    url_id = "1gJW9kgdfnVU1FzlzijRiup_8gkJNit82"
    zip_path = "linkedin_dataset.zip"
    gdown.download(id=url_id, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_FOLDER)
    os.remove(zip_path)
    return os.path.join(DATA_FOLDER, "linkedin_dataset.csv")

DATA_FILE = download_and_extract_dataset()

# Resume Parsing Dummy
@st.cache_data
def parse_resume(file_path):
    return {
        "name": "John Doe",
        "email": "john@example.com",
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "experience": "2 years at ABC Corp",
        "education": "B.Sc. in Computer Science"
    }

# Generate Questions
def generate_questions(role):
    return [
        f"What are the key responsibilities of a {role}?",
        f"How do you stay updated with trends in {role}?",
        f"Explain a project where you demonstrated {role}-related skills."
    ]

# Score Calculation
def calculate_score(answers, keywords):
    score = sum(1 for ans, kw in zip(answers, keywords) if kw.lower() in ans.lower())
    return round((score / len(keywords)) * 100, 2)

# Jooble Integration
def fetch_job_recommendations(role, location="India"):
    try:
        url = f"https://jooble.org/api/{JOOBLE_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {"keywords": role, "location": location, "page": 1, "searchMode": 1}
        response = requests.post(url, json=payload, headers=headers)
        jobs = response.json().get("jobs", [])
        return jobs[:5]
    except Exception as e:
        st.error(f"Jooble Error: {e}")
        return []

# PySpark Feedback
def generate_bigdata_feedback(user_skills):
    try:
        spark = SparkSession.builder.appName("LinkedInAnalysis").getOrCreate()
        df = spark.read.csv(DATA_FILE, header=True, inferSchema=True)
        df = df.dropna(subset=["Skills"])
        skill_match_count = df.rdd.map(lambda row: len(set(user_skills).intersection(set(row["Skills"].split(','))))).sum()
        avg_match = round(skill_match_count / df.count(), 2)
        spark.stop()
        return f"Your skills match an average of {avg_match} LinkedIn profiles."
    except Exception as e:
        return f"Error analyzing LinkedIn data: {e}"

# Email Feedback
def send_feedback_email(to_email, name, role, score, feedback_note):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = f"SmartHire AI Interview Report – {role}"
        body = f"""Hello {name},\n\nThank you for your interview.\nScore: {score}%\n\nInsights:\n{feedback_note}\n\nBest,\nSmartHire AI"""
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        st.warning(f"Email error: {e}")

# Streamlit App
st.title("SmartHire AI - Career Interview Assistant")

if "resume" not in st.session_state:
    st.session_state.resume = {}

with st.sidebar:
    st.header("Admin Login")
    admin_user = st.text_input("Email")
    admin_pass = st.text_input("Password", type="password")
    if st.button("View Reports"):
        if admin_user == ADMIN_EMAIL and admin_pass == ADMIN_PASSWORD:
            if os.path.exists(REPORTS_FILE):
                df = pd.read_csv(REPORTS_FILE, names=["Name", "Email", "Role", "Score", "Date"])
                st.dataframe(df)
                st.download_button("Download Reports CSV", df.to_csv(index=False), "interview_reports.csv")
            else:
                st.info("No reports yet.")
        else:
            st.warning("Invalid credentials.")

st.subheader("Upload Your Resume")
input_lang = st.selectbox("Input Language", ["English", "Hindi", "French"])
output_lang = st.selectbox("Output Language", ["English", "Hindi", "French"])
file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if file:
    file_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    resume_data = parse_resume(file_path)
    st.session_state.resume = resume_data
    st.success("Resume parsed successfully!")
    st.write(resume_data)

    role = st.text_input("Enter your desired role")
    if role:
        questions = generate_questions(role)
        st.session_state.questions = questions
        st.write("### Interview Questions")
        answers = []
        for i, q in enumerate(questions):
            ans = st.text_area(f"{i+1}. {q}", key=f"answer_{i}")
            answers.append(ans)

        if st.button("Submit Interview"):
            score = calculate_score(answers, ["responsibilities", "trends", "project"])
            feedback_note = generate_bigdata_feedback(resume_data['skills'])
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            with open(REPORTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([resume_data['name'], resume_data['email'], role, score, now])
            send_feedback_email(resume_data['email'], resume_data['name'], role, score, feedback_note)
            jobs = fetch_job_recommendations(role)
            st.success(f"Interview Score: {score}%")
            st.info(feedback_note)
            st.write("### Job Recommendations")
            for job in jobs:
                st.write(f"**{job.get('title')}** at {job.get('company')}\n{job.get('snippet')}\n[View Job]({job.get('link')})")
