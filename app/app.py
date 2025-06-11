import streamlit as st
import logging
import datetime
import os
from typing import Optional

from llm import expand_prompt, detect_intent_ollama
from text2image import bring_idea_to_life
from memory_manager import MemoryManager
from core.stub import Stub


# --- Setup Main App Logger ---
# This logger will capture general application activity and can log to console and a main file.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Ensure handlers are added only once to the main logger
if not logger.handlers: # This check is crucial for Streamlit reruns
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Added console handler to main logger.")

    # Create a file handler for general app activity
    # This will save logs to 'app_activity.log' in the 'Logs' directory
    LOGS_DIR = "./Logs"
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        logger.info(f"Created main logs directory: {LOGS_DIR}")
        
    main_file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'app_activity.log'))
    main_file_handler.setFormatter(formatter)
    logger.addHandler(main_file_handler)
    logger.info(f"Added file handler to main logger: {os.path.join(LOGS_DIR, 'app_activity.log')}")


# --- Session-specific Logging for Generation Details ---
# This block ensures a unique log file for EACH Streamlit user session.
# It runs only once when a new user session begins.
if 'session_logger' not in st.session_state:
    session_log_filename = f"generation_details_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Defining a dedicated subdirectory for session logs
    SESSION_LOGS_DIR = os.path.join(LOGS_DIR, "sessions") 
    if not os.path.exists(SESSION_LOGS_DIR):
        os.makedirs(SESSION_LOGS_DIR)
        logger.info(f"Created session-specific logs directory: {SESSION_LOGS_DIR}")
        
    session_log_file_path = os.path.join(SESSION_LOGS_DIR, session_log_filename)

    # Create a new logger instance unique to this session
    session_logger = logging.getLogger(session_log_filename) 
    session_logger.setLevel(logging.INFO)

    # Add a file handler for this session's log file
    # This `if not session_logger.handlers:` check is critical for Streamlit
    # to prevent adding multiple handlers on subsequent reruns within the same session.
    if not session_logger.handlers: 
        session_file_handler = logging.FileHandler(session_log_file_path, mode='a')
        session_file_handler.setFormatter(formatter)
        session_logger.addHandler(session_file_handler)
        # Prevent propagation to the root logger to avoid duplicate entries in app_activity.log
        session_logger.propagate = False 
        logger.info(f"Initialized session-specific logger for file: {session_log_file_path}")
    
    # Store the configured session_logger in Streamlit's session state.
    # This ensures the same logger instance (and thus the same unique log file) is used
    # throughout the current user's session.
    st.session_state.session_logger = session_logger

# --- Global/Session State Initialization ---
if 'memory_manager' not in st.session_state:
    try:
        st.session_state.memory_manager = MemoryManager(ollama_model_name="llama3", logger_instance=st.session_state.session_logger)
        logger.info("MemoryManager initialized in Streamlit session state.")
    except Exception as e:
        logger.error(f"Failed to initialize MemoryManager: {e}")
        st.session_state.memory_manager = None
        st.error("Error initializing memory. Memory features will be unavailable.")

if 'stub' not in st.session_state:
    st.session_state.openfabric_app_ids = [
        "c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network"
    ]
    st.session_state.stub = Stub(st.session_state.openfabric_app_ids)
    logger.info(f"Openfabric Stub initialized with IDs: {st.session_state.openfabric_app_ids}")


OUTPUT_DIR = "OutputImage"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

# --- Initialize session state for displaying generated images/messages ---
if 'generated_outputs' not in st.session_state:
    st.session_state.generated_outputs = []

# --- Initialize user_prompt_input if it doesn't exist ---
# This ensures it's initialized BEFORE the text_area widget is instantiated.
if 'user_prompt_input' not in st.session_state:
    st.session_state.user_prompt_input = ""


def generate_creative_asset(user_prompt: str):
    mm = st.session_state.memory_manager
    stub_instance = st.session_state.stub
    # Retrieve the session-specific logger
    session_logger = st.session_state.session_logger 

    # Using a placeholder for messages to ensure they appear sequentially in the UI
    message_placeholder = st.empty()
    progress_bar = st.progress(0)

    message_placeholder.info("Processing your creative request...")

    # Step 0: Detect intent (remix or new creation)
    intent_type = "new_generation" # Default if memory is not available
    if mm:
        try:
            session_logger.info(f"Analyzing user prompt for intent: '{user_prompt[:50]}...'")
            intent_type = detect_intent_ollama(user_prompt)
            session_logger.info(f"Detected intent: {intent_type}")
            progress_bar.progress(20)
        except Exception as e:
            session_logger.error(f"Error detecting intent: {e}") 
            message_placeholder.warning("Could not detect intent, proceeding with new generation.")
            intent_type = "new_generation"

    final_prompt_for_generation = user_prompt
    recalled_creation = None

    # --- Handle Remix Intent ---
    if intent_type == "remix" and mm:
        message_placeholder.info("Remix intent detected. Attempting to recall past creations.")
        session_logger.info("Remix intent detected. Attempting to recall past creations.")

        try:
            # Search both memories
            short_term_results = mm.search_short_term_memory(user_prompt, n_results=1)
            long_term_results = mm.search_long_term_memory(user_prompt, n_results=1)

            # Select the best match from either memory
            recalled_creation, similarity_score = mm.select_best_memory_match(short_term_results, long_term_results)

            if recalled_creation:
                recalled_expanded_prompt = recalled_creation.get('expanded_prompt', recalled_creation.get('user_prompt'))
                reference_prompt = recalled_creation.get('user_prompt', '')[:50]

                final_prompt_for_generation = (
                    f"User wants to remix. Here's a detailed description of a past creation: "
                    f"'{recalled_expanded_prompt}'. "
                    f"Now, based on the user's new request: '{user_prompt}', "
                    f"creatively combine these elements into a single, vivid, and detailed artistic description suitable for image generation. "
                    f"Ensure the new description incorporates the modifications or references from the user's latest input."
                )

                message_placeholder.write(f"**Remixing based on:** \"{reference_prompt}...\"")
                session_logger.info(f"Constructed remix prompt for LLM expansion: '{final_prompt_for_generation[:100]}...'")
            else:
                message_placeholder.info("No relevant past creation found to remix. Treating as a new creation.")
                session_logger.info("Fallback to new generation: no relevant match found.")
                intent_type = "new_generation"

        except Exception as e:
            session_logger.error(f"Error during remix memory recall: {e}")
            message_placeholder.error("An error occurred during memory recall. Proceeding with new generation.")
            intent_type = "new_generation"


    progress_bar.progress(40)


    # Step 1: Expand prompt using local LLM
    try:
        expanded_prompt_by_llm = expand_prompt(final_prompt_for_generation)
        session_logger.info(f"LLM generated expanded prompt: {expanded_prompt_by_llm[:150]}...") 
        progress_bar.progress(60)
    except Exception as e:
        session_logger.error(f"Error expanding prompt: {e}") 
        message_placeholder.error("Failed to expand prompt using LLM. Please check your Ollama setup.")
        progress_bar.empty()
        return


    # Step 2: Generate Image Data using the Text-to-Image app
    image_data: Optional[bytes] = None
    try:
        image_data = bring_idea_to_life(expanded_prompt_by_llm, stub_instance)
        session_logger.info(f"Image generation initiated. Data type: {type(image_data)}") 
        progress_bar.progress(80)
    except Exception as e:
        session_logger.error(f"Error generating image: {e}") 
        message_placeholder.error("Failed to generate image from prompt. Please check the Text-to-Image app connection.")
        progress_bar.empty()
        return

    output_image_filename: Optional[str] = None
    if image_data and isinstance(image_data, bytes):
        image_name = f"generated_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        output_image_filename = os.path.join(OUTPUT_DIR, image_name)
        try:
            with open(output_image_filename, 'wb') as f:
                f.write(image_data)
            session_logger.info(f"Generated image saved as: {output_image_filename}")
            
            st.session_state.generated_outputs.append({
                'type': 'image',
                'content': image_data,
                'caption': f"Image for: '{user_prompt}'"
            })
            st.session_state.generated_outputs.append({
                'type': 'message',
                'content': "Image generated successfully!"
            })
            
            if mm:
                session_logger.info(f"--- Creative Asset Log ---")
                session_logger.info(f"User Prompt: {user_prompt}")
                session_logger.info(f"Expanded Prompt: {expanded_prompt_by_llm}")
                session_logger.info(f"Generated Image Path: {output_image_filename}")
                session_logger.info(f"--------------------------")

                saved_id = mm.save_creation(
                    user_prompt=user_prompt,
                    expanded_prompt=expanded_prompt_by_llm
                )
                if saved_id:
                    session_logger.info(f"Creation successfully saved to memory with ID: {saved_id}")
                    st.session_state.generated_outputs.append({
                        'type': 'message',
                        'content': f"Creative brief saved to memory (ID: {saved_id})"
                    })
                else:
                    session_logger.error("Failed to save creation to memory.")
                    st.session_state.generated_outputs.append({
                        'type': 'message',
                        'content': "Failed to save creation to memory."
                    })
            else:
                st.session_state.generated_outputs.append({
                    'type': 'message',
                    'content': "MemoryManager not available. Skipping saving to memory."
                })

        except Exception as e:
            session_logger.error(f"Failed to save and display image: {e}")
            st.session_state.generated_outputs.append({
                'type': 'message',
                'content': f"Failed to save or display image. Error: {e}"
            })
    else:
        st.session_state.generated_outputs.append({
            'type': 'message',
            'content': "Image generation failed or returned invalid data."
        })
        session_logger.error("Image generation failed or returned invalid data.")

    progress_bar.empty()
    message_placeholder.empty()


# --- Callback function to handle button click and clear input ---
def handle_generate_click():
    if st.session_state.user_prompt_input:
        generate_creative_asset(st.session_state.user_prompt_input)
        st.session_state.user_prompt_input = ""
    else:
        st.warning("Please enter a description to generate an image.")


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Creative Idea Generator",
    page_icon="✨",
    layout="centered"
)

st.title("✨ Creative Idea Generator")
st.markdown("Enter your idea below, and I'll bring it to life as an image!")

# --- Display generated outputs in a scrollable container ---
output_container = st.container(height=500, border=True)

with output_container:
    if not st.session_state.generated_outputs:
        st.info("Your generated images and messages will appear here.")
    else:
        for output in st.session_state.generated_outputs:
            if output['type'] == 'image':
                st.image(output['content'], caption=output['caption'], use_column_width=True)
            elif output['type'] == 'message':
                st.write(output['content'])
            st.markdown("---")


# --- Input section at the bottom ---
st.markdown("---")

user_input_col, button_col = st.columns([4, 1])

with user_input_col:
    # `value` is set to st.session_state.user_prompt_input for controlled input
    # `on_change` is not used here as we clear after button click
    st.text_area(
        "Describe your creative vision:",
        placeholder="e.g., A fantastical forest with glowing mushrooms and a mischievous pixie. (Scroll above to see results)",
        height=100,
        key="user_prompt_input",
        label_visibility="collapsed",
        value=st.session_state.user_prompt_input
    )

with button_col:
    st.write("")
    st.write("")
    st.button(
        "Generate",
        type="primary",
        use_container_width=True,
        on_click=handle_generate_click
    )