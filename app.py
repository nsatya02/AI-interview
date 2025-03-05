from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Streamlit page configuration
st.set_page_config(
    page_title="AI Interview Coach",
    layout="wide",
    initial_sidebar_state="expanded",
)
model = genai.GenerativeModel("model-name")

# Function to get a question from Gemini based on type, company, role, topic, and difficulty
def get_interview_question(interview_type, company, role, topic, difficulty):
    prompt = f"Generate a {difficulty} level {interview_type} interview question focusing on {topic} for a {role} position at {company}."
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "No question generated. Try again."

# Function to get a response from Gemini
def get_gemini_response(answer):
    try:
        response = model.generate_content(f"Evaluate this interview response: {answer}")
        return response.text.strip() if response.text else "No response available. Try again."
    except Exception as e:
        return f"Error: {e}"

# UI Enhancements
st.markdown(
    """
    <style>
        .title-text { text-align: center; font-size: 32px; font-weight: bold; color: #2E86C1; margin-bottom: 20px; }
        .stButton>button { width: 60%; padding: 6px; border-radius: 8px; background-color: #4A90E2; color: white; font-size: 12px; display: block; margin: 10px auto; }
        .stTextInput>div>div>input, .stTextArea>div>textarea { font-size: 16px; padding: 10px; border-radius: 8px; }
        .question-box { background-color: #F8F9F9; padding: 15px; border-radius: 10px; border-left: 5px solid #4A90E2; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title-text'>📝 AI Interview Trainer</div>", unsafe_allow_html=True)

# Layout for left sidebar (1/4) and main content (3/4)
col1, col2 = st.columns([1, 3])

with col1:
    role_name = st.text_input("Enter your role (e.g., Software Engineer)")
    company_name = st.text_input("Target Company")
    difficulty_levels = ["Easy", "Medium", "Hard"]
    difficulty = st.selectbox("Select Question Difficulty", difficulty_levels)
    topics = ["Data Structures", "Algorithms", "System Design", "Machine Learning", "Cybersecurity","Statistics"]
    topic = st.selectbox("Select Topic", topics)
    interview_types = ["Technical", "Behavioral", "HR"]
    interview_type = st.selectbox("Select Interview Type", interview_types)
    num_questions = st.slider("Number of Questions", 1, 10, 5)

with col2:
    if st.button("Generate Next Question"):
        if not company_name or not role_name:
            st.warning("Please enter both company name and role to generate a question.")
        else:
            st.session_state["question"] = get_interview_question(interview_type, company_name, role_name, topic, difficulty)
            st.markdown(f"<div class='question-box'><strong>Question {num_questions}:</strong> {st.session_state['question']}</div>", unsafe_allow_html=True)
    
    user_response = st.text_area("Your Answer:", height=150)
    if st.button("Submit Answer"):
        if "question" not in st.session_state:
            st.warning("Please generate a question first!")
        elif not user_response.strip():
            st.warning("Please enter your response before submitting.")
        else:
            feedback = get_gemini_response(user_response)
            st.subheader("AI Feedback:")
            st.markdown(f"<div class='question-box'>{feedback}</div>", unsafe_allow_html=True)

st.markdown("---")
#st.markdown("<div style='text-align: center; font-size: 18px; font-weight: bold; color: #2E86C1;'>Practice real-world interview questions and get AI-powered feedback! 🚀</div>", unsafe_allow_html=True)
