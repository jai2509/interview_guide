# app.py

import streamlit as st
import pdfplumber
import spacy
import speech_recognition as sr
import requests
import json
import os
from email.message import EmailMessage
import smtplib
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract skills from resume
def extract_skills_from_resume(file):
    skills = []
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        doc = nlp(text)
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                skills.append(token.text.lower())
    return list(set(skills))

# Call Groq API to generate interview questions
def generate_interview_questions(skills, role):
    prompt = f"""
You are an expert AI interviewer.
Generate 3 professional and technical interview questions for a candidate applying for the role of {role}.
The candidate has the following skills: {', '.join(skills)}.
Questions should test the candidate's knowledge deeply.
Only output the questions without numbering or extra text.
"""
    groq_api_key = st.secrets["GROQ_API_KEY"]
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    result = response.json()

    generated_text = result['choices'][0]['message']['content']
    questions = generated_text.strip().split('\n')
    questions = [q.strip("-").strip() for q in questions if q.strip()]
    return questions[:3]  # Return first 3 questions

# Record and transcribe answer
def record_answer():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Recording... Speak now!")
        audio = r.listen(source, timeout=10, phrase_time_limit=60)
        st.success("Recording complete!")
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, could not understand your answer."
    except sr.RequestError:
        return "Speech recognition service error."

# Analyze answer for feedback
def analyze_answer(answer, expected_keywords):
    feedback = ""
    answer_lower = answer.lower()
    
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    coverage = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0

    word_count = len(answer.split())

    if word_count < 30:
        feedback += "- Try to explain in more detail.\n"
    elif word_count > 150:
        feedback += "- Keep answers more concise.\n"
    else:
        feedback += "- Good explanation length.\n"

    if coverage > 0.7:
        feedback += "- Covered most key points!\n"
    elif coverage > 0.4:
        feedback += "- Covered some key points, but missed a few.\n"
    else:
        feedback += "- Missed important concepts, please revise.\n"

    return feedback

# Send feedback report via email
def send_email_report(receiver_email, report_text):
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    msg = EmailMessage()
    msg['Subject'] = 'Your AI Interview Feedback Report'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(report_text)

    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Interview Expert", page_icon="🎤", layout="centered")
    st.title("🎤 AI Interview Expert")
    st.subheader("Upload Resume > Select Role > Speak Answers > Get AI Feedback")

    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

    if uploaded_file is not None:
        skills = extract_skills_from_resume(uploaded_file)
        st.success(f"Resume parsed successfully! {len(skills)} skills detected.")

        role = st.text_input("Enter your Target Role (example: Data Scientist, ML Engineer, etc.)")

        if st.button("Generate Interview Questions"):
            if role:
                questions = generate_interview_questions(skills, role)
                st.success("Interview questions generated successfully!")

                report_text = f"AI Interview Feedback Report for Role: {role}\n\n"

                for idx, question in enumerate(questions):
                    st.markdown(f"### Question {idx+1}: {question}")
                    
                    if st.button(f"Record Answer for Q{idx+1}"):
                        answer = record_answer()
                        st.markdown(f"**Your Answer:** {answer}")

                        expected_keywords = re.findall(r'\w+', question)
                        feedback = analyze_answer(answer, expected_keywords)
                        report_text += f"Question {idx+1}: {question}\n"
                        report_text += f"Answer: {answer}\n"
                        report_text += f"Feedback: {feedback}\n\n"

                email = st.text_input("Enter your Email to receive Full Feedback Report:")

                if st.button("Send Feedback Report"):
                    if email:
                        send_email_report(email, report_text)
                        st.success("Feedback Report sent successfully!")
                    else:
                        st.error("Please enter a valid email address!")
            else:
                st.error("Please enter a target role!")

if __name__ == "__main__":
    main()
