import streamlit as st
from dotenv import load_dotenv
import os
import io
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# --- New Imports for TTS/STT ---
from gtts import gTTS
import speech_recognition as sr
# import time # Not strictly needed for this fix, but can be useful

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AVATAR_IMAGE_PATH = "/Users/satyanarasala/Desktop/Interview-final/AI-interview/14b882fb-9a2f-4add-b549-b8f1916f8ebf.JPG"


# --- Helper Functions ---

@st.cache_data(show_spinner="Extracting text from PDF...")
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_bytes = pdf_file.getvalue()
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)
        text = ""
        if not reader.pages:
             st.warning("PDF seems empty or could not be read.")
             return None
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                 text += page_text + "\n"
        # Check if text is effectively empty after stripping
        processed_text = text.strip()
        if not processed_text:
             st.warning("Could not extract meaningful text from the PDF. It might be image-based or empty.")
             return None
        return processed_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# @st.cache_resource # Caching LLM chain can be tricky with memory and state resets
def initialize_llm_chain(_resume_text):
    """Initializes the LangChain ConversationChain with Groq LLM."""
    template = """You are an expert AI interviewer conducting a screening interview.
    Your goal is to assess the candidate's suitability based SOLELY on their resume and the conversation.
    Be professional, insightful, and ask relevant questions. Keep your responses concise and focused on asking the next question or reacting briefly.
    DO NOT greet the user again after the first message. Focus on the interview questions. Start directly with the first question after the initial (implied) greeting.

    Candidate's Resume Summary:
    ---
    {resume_summary}
    ---

    Current Conversation:
    {history}

    Human: {input}
    AI Interviewer:"""

    prompt = PromptTemplate(
        input_variables=["history", "input", "resume_summary"],
        template=template
    )

    resume_summary_text = _resume_text if _resume_text else "No resume summary provided."
    # Use partial correctly on the prompt object
    prompt = prompt.partial(resume_summary=resume_summary_text)

    try:
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192", # Or your preferred model
        )
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        st.warning("Ensure Groq API key is set correctly in your environment.")
        return None

    # return_messages=True is needed for chat_message component type checking
    memory = ConversationBufferMemory(human_prefix="Human", ai_prefix="AI Interviewer", memory_key="history", return_messages=True)

    conversation = ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False # Set to True for debugging Langchain steps
        )

    return conversation

import base64 # <<< Import base64

# Keep caching if returning bytes
@st.cache_data(show_spinner=False)
def text_to_speech(text):
    """Converts text to speech using gTTS and returns audio bytes."""
    # st.write(f"DEBUG: TTS Start for text: '{text[:50]}...'")
    if not text or not isinstance(text, str) or text.strip() == "":
        st.warning("TTS Warning: Received empty or invalid text. Skipping audio generation.")
        return None
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()

        if not audio_bytes:
             st.warning("TTS Warning: gTTS generated empty audio bytes.")
             return None
        # print(f"DEBUG: TTS generated {len(audio_bytes)} bytes.") # Optional console debug
        return audio_bytes # <<< RETURN BYTES
    except Exception as e:
        st.error(f"TTS Error: Failed to generate audio. Details: {e}", icon="🚨")
        print(f"TTS Generation Error: {e}")
        return None


def speech_to_text(recognizer, microphone):
    """Captures audio from the microphone and transcribes it. Returns (transcript, status_message)."""
    if not isinstance(recognizer, sr.Recognizer):
        st.error("Speech Recognition Recognizer not initialized.")
        return None, "Error: Recognizer not ready."
    if not isinstance(microphone, sr.Microphone):
        st.error("Speech Recognition Microphone not initialized.")
        return None, "Error: Microphone not ready."

    transcript = None
    status_message = ""
    with microphone as source:
        st.info("Adjusting for ambient noise...")
        try:
            # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Short adjustment
            recognizer.dynamic_energy_threshold = True # Often better than fixed adjustment
            st.info("Listening... Speak clearly (up to 30 seconds).")
            # Increased timeout and phrase_time_limit for potentially longer answers
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=30) # Increased timeouts
        except sr.WaitTimeoutError:
            st.warning("No speech detected within the time limit.")
            status_message = "Warning: No speech detected."
            return None, status_message
        except Exception as e:
            st.error(f"Error during audio listening: {e}")
            status_message = f"Error: Listening failed ({e})."
            return None, status_message

    st.info("Processing speech...")
    try:
        # Use Google Web Speech API for transcription
        transcript = recognizer.recognize_google(audio)
        st.success("Transcription successful!")
        status_message = "Success: Transcribed."
        return transcript, status_message
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        status_message = f"Error: API request failed ({e})."
        return None, status_message
    except sr.UnknownValueError:
        st.warning("Google Speech Recognition could not understand audio.")
        status_message = "Warning: Could not understand audio."
        return None, status_message
    except Exception as e:
        st.error(f"Error during speech recognition: {e}")
        status_message = f"Error: Transcription failed ({e})."
        return None, status_message

# --- Function to Process Input (Centralized Logic) ---
def process_user_input(user_input):
    """Adds user input to history, gets AI response, adds to history, generates TTS file path."""
    # Clear previous AI audio path *before* generating new response
    st.session_state.current_ai_audio_path = None # Use the new state variable name
    

    # Generate AI response
    ai_response_text = None
    with st.spinner("AI is thinking..."):
        try:
            if st.session_state.conversation_chain:
                ai_response_text = st.session_state.conversation_chain.predict(input=user_input)

                if not ai_response_text or not isinstance(ai_response_text, str) or ai_response_text.strip() == "":
                     st.warning("AI Warning: Received empty or invalid response from LLM.")
                     st.session_state.chat_history.append(AIMessage(content="[AI response was empty or invalid]"))
                     st.session_state.current_ai_audio_bytes = None # Ensure no path
                else:
                    st.session_state.chat_history.append(AIMessage(content=ai_response_text))
                    
                    # Generate TTS and get the file path
                    audio_bytes_result = text_to_speech(ai_response_text) # Function now returns path
                    st.session_state.current_ai_audio_bytes = audio_bytes_result # Store the path

            else:
                 st.error("Conversation Chain not initialized...")
                 st.session_state.chat_history.append(AIMessage(content="[Error: AI Conversation Chain not ready]"))
                 st.session_state.current_ai_audio_bytes = None

        except Exception as e:
            error_msg = f"Error getting AI response: {e}"
            st.error(error_msg, icon="🔥")
            print(f"LLM Prediction Error: {e}")
            st.session_state.chat_history.append(AIMessage(content=f"[Error during AI response generation: {e}]"))
            st.session_state.current_ai_audio_bytes = None

    # Clean up the STT processing flag if it was set
    if "stt_triggered_processing" in st.session_state:
        del st.session_state.stt_triggered_processing
       

# --- Streamlit App ---
st.set_page_config(page_title="AI Interviewer (Groq + Avatar)", layout="wide")
st.title("🤖 AI Interviewer Bot")
st.markdown("Upload your resume, and I'll ask you questions. Respond via text or voice!")

# --- Session State Initialization ---
# Use functions for complex initializations to avoid errors if libraries fail
def init_recognizer():
    try:
        return sr.Recognizer()
    except Exception as e:
        st.error(f"Failed to initialize Speech Recognizer: {e}. Please ensure microphone access and necessary libraries (like PyAudio) are set up.")
        return None

def init_microphone():
    try:
        # Check for microphones
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
             st.warning("No microphones found by SpeechRecognition. Ensure a microphone is connected and drivers are installed.")
             return None
        # You could potentially let the user choose a microphone if multiple are found
        # st.write("Available Microphones:", mic_list) # For debugging
        return sr.Microphone() # Use default microphone
    except AttributeError:
         st.error("Could not find PyAudio. Ensure it's installed (`pip install pyaudio` or `conda install pyaudio`). Microphone functions disabled.")
         return None
    except Exception as e:
         st.error(f"Failed to initialize Microphone: {e}. Check microphone permissions and drivers. Microphone functions disabled.")
         return None

# Initialize only if not already in state
if "resume_text" not in st.session_state: st.session_state.resume_text = None
if "conversation_chain" not in st.session_state: st.session_state.conversation_chain = None
if "chat_history" not in st.session_state: st.session_state.chat_history = [] # Stores Langchain Message objects
if "interview_started" not in st.session_state: st.session_state.interview_started = False
if "processing_done" not in st.session_state: st.session_state.processing_done = False # Tracks if PDF processing is complete for the current upload
if "api_key_valid" not in st.session_state: st.session_state.api_key_valid = bool(GROQ_API_KEY)
if "current_ai_audio" not in st.session_state: st.session_state.current_ai_audio = None # Stores bytes of latest AI response TTS
if "stt_transcript_result" not in st.session_state: st.session_state.stt_transcript_result = None # Holds the latest STT result (transcript, status_message) tuple
if "recognizer" not in st.session_state: st.session_state.recognizer = init_recognizer()
if "microphone" not in st.session_state: st.session_state.microphone = init_microphone()
if "stt_in_progress" not in st.session_state: st.session_state.stt_in_progress = False # Flag to prevent multiple STT runs and update button label


# --- Sidebar ---
with st.sidebar:
    st.header("Setup")
    # Use a unique key and on_change to reset state when a new file is uploaded
    uploaded_file = st.file_uploader(
        "1. Upload Resume (PDF)",
        type="pdf",
        key="pdf_uploader_key", # Unique key
        # Reset key states when a new file is chosen (even before it's fully uploaded/processed)
        on_change=lambda: st.session_state.update(
            processing_done=False,
            interview_started=False,
            chat_history=[],
            conversation_chain=None,
            current_ai_audio=None,
            stt_transcript_result=None,
            resume_text=None # Also clear previous resume text
            )
    )

    # Process PDF only if a file is uploaded AND processing isn't marked as done yet for this file
    if uploaded_file and not st.session_state.processing_done:
        with st.spinner("Processing Resume..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text:
                st.session_state.resume_text = extracted_text
                st.session_state.processing_done = True # Mark processing as done for this file
                st.success("Resume Processed!")
                # Don't rerun here, let the main flow continue
            else:
                # Error/warning handled within extract_text_from_pdf
                st.session_state.resume_text = None
                st.session_state.processing_done = False # Ensure it's False if extraction failed

    # API Key Check Display
    if not st.session_state.api_key_valid:
        st.error("Groq API Key not found (GROQ_API_KEY environment variable).")
    else:
        # Show start button only if resume is processed and API key is valid
        if st.session_state.processing_done:
            if not st.session_state.interview_started:
                if st.button("2. Start Interview", key="start_interview_btn", type="primary"):
                    with st.spinner("Initializing Interview..."):
                        st.session_state.conversation_chain = initialize_llm_chain(st.session_state.resume_text)
                        if st.session_state.conversation_chain:
                            st.session_state.interview_started = True
                            st.session_state.chat_history = [] # Clear history just in case
                            st.session_state.current_ai_audio = None
                            st.session_state.stt_transcript_result = None

                            # Generate the first AI message
                            # Use a simple primer for the AI via the processing function
                            initial_primer = "Start the interview by asking the first question based on the resume."
                            process_user_input(initial_primer)

                            # Remove the initial primer message from history if you don't want it shown
                            if st.session_state.chat_history and isinstance(st.session_state.chat_history[0], HumanMessage) and st.session_state.chat_history[0].content == initial_primer:
                                 st.session_state.chat_history.pop(0)

                            st.rerun() # Rerun to display the first AI message and audio player
                        else:
                            st.error("Failed to initialize AI Conversation Chain. Check API key and Groq status.")
                            # Keep interview_started as False

            # Show Restart button only if interview has started
            elif st.session_state.interview_started:
                 if st.button("Restart Interview", key="restart_interview_btn"):
                    # Reset states for a restart
                    st.session_state.interview_started = False
                    st.session_state.chat_history = []
                    st.session_state.conversation_chain = None # Will be re-initialized on next start
                    st.session_state.current_ai_audio = None
                    st.session_state.stt_transcript_result = None
                    st.session_state.stt_in_progress = False # Reset STT flag too
                    # Keep resume processed (processing_done=True) unless a new file is uploaded
                    st.rerun()

# --- Main Interview Area ---

# Check if STT transcription just finished and needs processing
# This block runs at the start of a rerun if the flag was set in the *previous* run
if st.session_state.get("stt_triggered_processing"):
    transcript_data = st.session_state.stt_transcript_result
    if transcript_data and transcript_data[0]: # Check if transcript exists
        process_user_input(transcript_data[0]) # Process the transcript
        # The flag `stt_triggered_processing` is cleared inside process_user_input
        st.rerun() # Rerun IMMEDIATELY to display the new state (human msg + AI response + audio)
    else:
        # If STT finished but didn't yield a transcript (e.g., timeout, unknown value),
        # just clear the trigger flag and let the rest of the UI render normally
        # (the status message will be shown below).
        del st.session_state.stt_triggered_processing


# Display interview UI only if started and chain is ready
if st.session_state.interview_started and st.session_state.conversation_chain:

    # --- Avatar and Current AI Audio ---
    col1, col2 = st.columns([1, 4])

    with col1:
        # ... (avatar rendering logic) ...

        # <<< Display audio player using HTML <audio> and Data URI >>>
        audio_bytes = st.session_state.get("current_ai_audio_bytes") # Use .get

        if audio_bytes:
            try:
                # st.write(f"DEBUG: Have {len(audio_bytes)} bytes for HTML audio.")
                # 1. Base64 encode the bytes
                b64 = base64.b64encode(audio_bytes).decode("utf-8")

                # 2. Create the Data URI string
                #    MIME type for MP3 is 'audio/mpeg'
                data_uri = f"data:audio/mpeg;base64,{b64}"

                # 3. Create the HTML audio tag
                audio_html = f"""
                <audio controls="controls" autobuffer="autobuffer" autoplay="autoplay">
                    <source src="{data_uri}" />
                </audio>
                """

                # 4. Render the HTML
                st.markdown(audio_html, unsafe_allow_html= True)
                st.caption("Listen to the interviewer (HTML Player)")

            except Exception as e:
                st.error(f"Error generating HTML audio player: {e}", icon="🎧")
                print(f"HTML Audio Player Error: {e}")

        elif st.session_state.chat_history and isinstance(st.session_state.chat_history[-1], AIMessage):
             # If the last message is AI but no audio bytes
             st.caption("(Audio unavailable for last message)")


    # --- Chat History ---
    with col2:
        st.subheader("Interview Conversation")
        # Use a container with height for scrollability
        chat_container = st.container(height=500) # Adjust height as needed
        with chat_container:
            if not st.session_state.chat_history:
                 st.info("The interview will appear here.")
            else:
                for message in st.session_state.chat_history:
                    # Ensure message has 'type' and 'content' attributes
                    if hasattr(message, 'type') and hasattr(message, 'content'):
                         # Check if type is 'ai' or 'human' before passing to chat_message
                         role = message.type if message.type in ['ai', 'human'] else 'system' # Default or handle other types
                         with st.chat_message(role): # Use message.type ('ai' or 'human')
                              st.markdown(message.content)
                    else:
                        # Handle potential unexpected items in chat_history
                        st.warning(f"Skipping display of unexpected message format: {type(message)}")


    # --- User Input Area (Text and STT) ---
    st.divider()

    # Display STT status message if one exists from the last attempt
    if st.session_state.stt_transcript_result:
        status_msg = st.session_state.stt_transcript_result[1] # Get the status message
        # Display only non-success messages prominently, clear after showing?
        # Let's show warnings/errors until the next attempt clears stt_transcript_result
        if "Warning:" in status_msg:
            st.warning(f"🎙️ Last recording: {status_msg}")
        elif "Error:" in status_msg:
             st.error(f"🎙️ Last recording: {status_msg}")
        


    # Input Controls: Place text input and STT button side-by-side
    input_col, button_col = st.columns([4, 1])

    with input_col:
        # Text Input - Processed when the user presses Enter
        user_input_text = st.chat_input("Type your answer here, or use the Record button...", key="user_text_input", disabled=st.session_state.stt_in_progress)
        if user_input_text:
            # The user typed something and pressed Enter
            process_user_input(user_input_text)
            # st.chat_input clears itself on Enter, and process_user_input leads to a rerun eventually
            st.rerun() # Rerun immediately to show the user's message and AI thinking spinner

    with button_col:
        # STT Input Button
        mic_ready = st.session_state.recognizer and st.session_state.microphone
        stt_disabled = st.session_state.stt_in_progress or not mic_ready
        button_label = "🔴 Recording..." if st.session_state.stt_in_progress else "🎤 Record"

        if st.button(button_label, key="record_btn", help="Click to record your answer (max 30s)", disabled=stt_disabled, use_container_width=True):
            if mic_ready:
                st.session_state.stt_in_progress = True
                st.session_state.stt_transcript_result = None # Clear previous result/status
                st.rerun() # Rerun to update button label to "Recording..." and disable text input
            else:
                st.error("Microphone not ready. Cannot record.")

    # Handle STT Recording state (this block runs *after* the button press causes a rerun and stt_in_progress is True)
    if st.session_state.stt_in_progress:
        # st.write("DEBUG: STT in progress, calling speech_to_text...")
        transcript, status = speech_to_text(st.session_state.recognizer, st.session_state.microphone)
        st.session_state.stt_transcript_result = (transcript, status) # Store tuple (transcript, status_message)
        st.session_state.stt_in_progress = False # Mark STT as finished

        # If transcription was successful, set a flag to process it on the *next* rerun cycle
        if transcript:
            st.session_state.stt_triggered_processing = True
        else:
            pass

        st.rerun() # Rerun again. This time, either the stt_triggered_processing block at the top will run,
                   # or the UI will simply redraw with the STT status message and enabled inputs.


# --- Initial State Guidance ---
elif not uploaded_file:
    st.info("Please upload your resume PDF in the sidebar to begin.")
elif not st.session_state.processing_done:
     # This state might occur if upload happened but processing failed
     st.warning("Could not process the uploaded PDF. Please ensure it contains selectable text and try uploading again.")
elif not st.session_state.api_key_valid:
     st.warning("Please configure your Groq API Key (set the GROQ_API_KEY environment variable) to proceed.")
elif st.session_state.processing_done and not st.session_state.interview_started:
    # Resume is processed, API key is valid, but interview hasn't started
    st.success("Resume processed successfully!")
    st.info("Click 'Start Interview' in the sidebar when ready.")
else:
    # Catch-all for unexpected states
    st.warning("Something went wrong. Please try reloading the page or restarting the interview.")