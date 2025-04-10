from dotenv import load_dotenv
import streamlit as st
import os
from groq import Groq
import time # Optional: for simulating delay if needed

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("API key not found. Please set the GROQ_API_KEY environment variable.")
    st.stop()

try:
    client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()


st.set_page_config(
    page_title="AI Interview Practice",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- AI Interaction Functions ---

def get_interview_questions(interview_type, role, topic, difficulty, num_questions):
    questions = []
    st.info(f"Generating {num_questions} question(s)... Please wait.") # User feedback
    progress_bar = st.progress(0)
    for i in range(num_questions):
        prompt = (
            f"Generate one unique {difficulty} level {interview_type} interview question "
            f"specifically focusing on the topic '{topic}' for a '{role}' position. "
            f"Do not include introductory phrases like 'Here is a question:'. Just provide the question text."
            # Optional: Add constraints if previous questions were similar
            # if questions:
            #     prompt += f"\nAvoid questions too similar to these: {'; '.join(questions[-3:])}" # Avoid last 3
        )
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150, # Limit question length
                temperature=0.7, # Adjust creativity
            )
            question_text = response.choices[0].message.content.strip()
            if question_text:
                questions.append(question_text)
            else:
                st.warning(f"Received empty question for iteration {i+1}. Skipping.")
                # You might want to retry here or just accept fewer questions
            progress_bar.progress((i + 1) / num_questions)
            time.sleep(0.1) # Small delay to allow progress bar update visibility
        except Exception as e:
            st.error(f"Error generating question {i+1}: {e}")
            # Decide how to handle errors: stop, skip, retry?
            # For now, let's add a placeholder and continue
            questions.append(f"Error generating question {i+1}. Please try generating again.")
            progress_bar.progress((i + 1) / num_questions) # Still update progress
            break # Stop generating further questions on error

    progress_bar.empty() # Remove progress bar after completion
    if not questions:
         st.warning("No questions were generated. Please check the inputs or try again.")
    return questions

# Function to get feedback from Groq
def get_ai_feedback(question, answer):
    if not answer or not answer.strip():
        return "Please provide an answer to evaluate."
    prompt = (
        f"You are an AI Interview Evaluator.\n"
        f"Interview Question: '{question}'\n"
        f"Candidate's Answer: '{answer}'\n\n"
        f"Please evaluate this answer based on clarity, correctness, completeness, and relevance to the question for the given role context (implicitly). "
        f"Provide constructive feedback, highlighting strengths and areas for improvement. Be specific and helpful."
        f"Keep the feedback concise (around 2-4 sentences)."
    )
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.5,
        )
        feedback_text = response.choices[0].message.content.strip()
        return feedback_text if feedback_text else "AI could not provide feedback for this answer."
    except Exception as e:
        st.error(f"Error getting AI feedback: {e}")
        return f"Error during feedback generation: {e}"

st.markdown(
    """
    <style>
        .title-text { text-align: center; font-size: 32px; font-weight: bold; color: #4A90E2; margin-bottom: 20px; }
        /* Make buttons more prominent and consistently sized */
        .stButton>button {
            width: 200px; /* Fixed width */
            padding: 10px 15px;
            border-radius: 8px;
            background-color: #4A90E2;
            color: white;
            font-size: 16px;
            display: block;
            margin: 15px auto; /* Center buttons */
            border: none; /* Remove default border */
            cursor: pointer; /* Add pointer cursor */
        }
        .stButton>button:hover {
            background-color: #357ABD; /* Darker shade on hover */
        }
         /* Style for disabled buttons */
        .stButton>button:disabled {
            background-color: #A0A0A0; /* Grey out disabled buttons */
            color: #D0D0D0;
            cursor: not-allowed;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
             font-size: 16px; padding: 10px; border-radius: 8px; border: 1px solid #ccc;
        }
        .question-box { background-color: #E8F0FE; padding: 20px; border-radius: 10px; border-left: 6px solid #4A90E2; margin-bottom: 20px; font-size: 18px; color: #333; }
        .feedback-box { background-color: #E8F8F5; padding: 15px; border-radius: 10px; border-left: 5px solid #1ABC9C; margin-top: 15px; font-size: 16px; color: #333; }
        .stAlert p { font-size: 16px; } /* Make warning/error text larger */
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title-text'>📝 AI Interview</div>", unsafe_allow_html=True)

# --- Initialize Session State ---
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "current_feedback" not in st.session_state:
    st.session_state.current_feedback = ""
if "current_answer" not in st.session_state:
    st.session_state.current_answer = "" # Store submitted answer
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False # Flag to track if questions generated


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Interview Setup")
    role_name = st.text_input("Enter your target role", "Software Engineer", key="role_input")
    # company_name = st.text_input("Target Company (Optional)") # Keep if you want to add it back to prompt
    difficulty_levels = ["Easy", "Medium", "Hard"]
    difficulty = st.selectbox("Select Question Difficulty", difficulty_levels, index=1, key="difficulty_select")
    topics = ["Data Structures", "Algorithms", "System Design", "Machine Learning", "Cloud Computing", "Behavioral", "Cybersecurity", "Statistics"]
    topic = st.selectbox("Select Topic", topics, key="topic_select")
    interview_types = ["Technical", "Behavioral", "Situational"] # Renamed HR to Situational/Behavioral
    interview_type = st.selectbox("Select Interview Type", interview_types, key="type_select")
    num_questions = st.slider("Number of Questions", 1, 10, 3, key="num_questions_slider") # Default 3

    # Generate Questions Button
    if st.button("Start Interview / Generate Questions", key="generate_btn"):
        if not role_name:
            st.warning("Please enter a target role.")
        else:
            # Reset state for a new interview
            st.session_state.questions = []
            st.session_state.current_question_index = 0
            st.session_state.show_feedback = False
            st.session_state.current_feedback = ""
            st.session_state.current_answer = ""
            st.session_state.interview_started = False # Reset flag

            # Fetch new questions
            # Pass company_name if using: get_interview_questions(..., company_name, ...)
            generated_questions = get_interview_questions(interview_type, role_name, topic, difficulty, num_questions)

            if generated_questions:
                st.session_state.questions = generated_questions
                st.session_state.interview_started = True # Set flag
                st.success(f"Generated {len(st.session_state.questions)} questions!")
                st.rerun() # Rerun to update the main area immediately
            else:
                 st.error("Failed to generate questions. Please try again.")

# --- Main Interview Area ---
if not st.session_state.interview_started:
    st.info("Please configure the interview settings in the sidebar and click 'Start Interview / Generate Questions'.")
elif not st.session_state.questions:
     st.warning("No questions available. Please generate questions using the sidebar.")
else:
    # Check if all questions have been answered
    if st.session_state.current_question_index >= len(st.session_state.questions):
        st.success("🎉 Congratulations! You have completed all the questions for this session.")
        #st.balloons()
        # Offer to start a new interview
        if st.button("Start New Interview", key="new_interview_btn"):
            # Reset state completely
            st.session_state.questions = []
            st.session_state.current_question_index = 0
            st.session_state.show_feedback = False
            st.session_state.current_feedback = ""
            st.session_state.current_answer = ""
            st.session_state.interview_started = False
            st.rerun()
    else:
        # Display Current Question
        current_q_index = st.session_state.current_question_index
        current_question = st.session_state.questions[current_q_index]

        st.subheader(f"Question {current_q_index + 1} of {len(st.session_state.questions)}")
        st.markdown(f"<div class='question-box'>{current_question}</div>", unsafe_allow_html=True)

        # Answer Area - Use a unique key based on index to clear it implicitly on next question
        user_answer = st.text_area(
            "Your Answer:",
            height=200,
            key=f"answer_area_{current_q_index}",
            # Disable if feedback is already shown for this question
            disabled=st.session_state.show_feedback
        )

        # Submit Answer Button - Only show if feedback hasn't been shown yet
        if not st.session_state.show_feedback:
            submit_button_disabled = not user_answer.strip() # Disable if text area is empty
            if st.button("Submit Answer", key=f"submit_btn_{current_q_index}", disabled=submit_button_disabled):
                st.session_state.current_answer = user_answer # Store the answer
                with st.spinner("Evaluating your answer..."):
                    feedback = get_ai_feedback(current_question, user_answer)
                st.session_state.current_feedback = feedback
                st.session_state.show_feedback = True
                st.rerun() # Rerun to display feedback and Next button

        # Display Feedback and Next Question Button
        if st.session_state.show_feedback:
            st.subheader("AI Feedback:")
            # Display the submitted answer for context
            st.markdown("**Your Answer:**")
            st.markdown(f"> {st.session_state.current_answer}") # Show what was evaluated
            st.markdown(f"<div class='feedback-box'>{st.session_state.current_feedback}</div>", unsafe_allow_html=True)

            # Next Question Button - Only appears after feedback is shown
            if st.button("Next Question", key=f"next_btn_{current_q_index}"):
                st.session_state.current_question_index += 1
                st.session_state.show_feedback = False # Reset for the next question
                st.session_state.current_feedback = ""  # Clear old feedback
                st.session_state.current_answer = ""   # Clear old answer
                # The text_area will clear automatically due to the key change on rerun
                st.rerun() # Rerun to show the next question

st.markdown("---")
