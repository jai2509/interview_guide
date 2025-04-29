# SmartHire AI - Streamlit Version (With Admin Dashboard & Visuals)

import streamlit as st
import pandas as pd
import os
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import zipfile
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
from groq import Groq

# Set Streamlit page config
st.set_page_config(
    page_title="SmartHire AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@smarthireai.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "data"
REPORTS_FILE = os.path.join(DATA_FOLDER, "interview_reports.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

@st.cache_resource
def download_and_extract_dataset():
    url = "https://drive.google.com/uc?id=1gJW9kgdfnVU1FzlzijRiup_8gkJNit82"
    zip_path = "linkedin_dataset.zip"
    gdown.download(url, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_FOLDER)
    os.remove(zip_path)
    return os.path.join(DATA_FOLDER, "linkedin_dataset.csv")

DATA_FILE = download_and_extract_dataset()

@st.cache_data
def parse_resume(file_path):
    try:
        reader = PdfReader(file_path)
        full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        lines = full_text.split("\n")
        email = next((line for line in lines if "@" in line and "." in line), "not_found@example.com")
        name = lines[0] if lines else "Unknown"
        return {
            "name": name.strip(),
            "email": email.strip(),
            "skills": [word for word in full_text.split() if word.istitle() and len(word) > 3][:10],
            "experience": "Extracted from resume",
            "education": "Extracted from resume",
            "full_text": full_text
        }
    except Exception as e:
        st.warning(f"Resume parsing failed: {e}")
        return {"name": "Unknown", "email": "error@example.com", "skills": [], "experience": "", "education": "", "full_text": ""}

def generate_questions_from_groq(resume_text):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""Based on the following resume, ask 3 technical and 2 behavioral interview questions:
\n{resume_text}"""
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768"
        )
        return chat_completion.choices[0].message.content.strip().split("\n")
    except Exception as e:
        return [f"Error generating questions: {e}"]

def calculate_score(answers, keywords):
    score = sum(1 for ans, kw in zip(answers, keywords) if kw.lower() in ans.lower())
    return round((score / len(keywords)) * 100, 2)

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

def generate_data_visuals(user_skills):
    try:
        df = pd.read_csv(DATA_FILE)
        st.subheader("LinkedIn Insights")

        if 'Skills' in df.columns:
            st.write("### Most Common Skills")
            df['Skills'] = df['Skills'].fillna('')
            all_skills = pd.Series(','.join(df['Skills']).split(',')).str.strip().value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=all_skills.values, y=all_skills.index, ax=ax)
            st.pyplot(fig)

        if 'Company' in df.columns:
            st.write("### Top Companies Hiring")
            top_companies = df['Company'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=top_companies.values, y=top_companies.index, ax=ax)
            st.pyplot(fig)

        if 'Region' in df.columns and 'Salary' in df.columns:
            st.write("### Average Salary by Region")
            region_salary = df.groupby('Region')['Salary'].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            region_salary.plot(kind='barh', ax=ax)
            st.pyplot(fig)

        return "Visual feedback generated."
    except Exception as e:
        return f"Error in visual feedback: {e}"

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

# --- Sidebar Role Selection ---
role = st.sidebar.selectbox("Select Mode", ["Candidate", "Admin"])

if role == "Admin":
    st.sidebar.subheader("🔐 Admin Login")
    input_email = st.sidebar.text_input("Admin Email")
    input_password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if input_email == ADMIN_EMAIL and input_password == ADMIN_PASSWORD:
            st.success("✅ Logged in as Admin")
            if os.path.exists(REPORTS_FILE):
                df_reports = pd.read_csv(REPORTS_FILE)
                st.subheader("📋 All Interview Reports")
                st.dataframe(df_reports)
            else:
                st.info("No reports available yet.")
        else:
            st.error("❌ Invalid admin credentials.")
else:
    st.title("SmartHire AI – Job Interview Assistant 🧠")
    st.write("Upload your resume to receive AI-generated interview questions, score analysis, job recommendations, and LinkedIn-based skill insights.")

    uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        resume_data = parse_resume(file_path)
        st.write("### Extracted Resume Data")
        st.json(resume_data)

        st.write("---")
        st.subheader("Interview Questions")
        questions = generate_questions_from_groq(resume_data['full_text'])
        answers = []
        for i, q in enumerate(questions):
            ans = st.text_input(f"Q{i+1}: {q}", key=f"q_{i}")
            answers.append(ans)

        if st.button("Submit Interview"):
            score = calculate_score(answers, resume_data['skills'])
            st.success(f"Interview Score: {score}%")

            feedback = generate_data_visuals(resume_data['skills'])
            send_feedback_email(resume_data['email'], resume_data['name'], resume_data['skills'][0] if resume_data['skills'] else 'Candidate', score, feedback)

            with open(REPORTS_FILE, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if os.path.getsize(REPORTS_FILE) == 0:
                    writer.writerow(["name", "email", "role", "score", "timestamp"])
                writer.writerow([resume_data['name'], resume_data['email'], resume_data['skills'][0] if resume_data['skills'] else 'N/A', score, datetime.now().isoformat()])

            st.success("Report sent and saved ✅")

            st.write("---")
            st.subheader("Job Recommendations via Jooble")
            jobs = fetch_job_recommendations(resume_data['skills'][0] if resume_data['skills'] else "developer")
            for job in jobs:
                st.markdown(f"**{job.get('title', 'N/A')}**  ")
                st.markdown(f"{job.get('company', 'Unknown')} | {job.get('location', 'N/A')}  ")
                st.markdown(f"[View Job Posting]({job.get('link')})")
