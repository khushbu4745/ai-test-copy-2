import os
import logging
import datetime
from typing import Dict, Optional

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub

from llm import expand_prompt, detect_intent_ollama
from text2image import bring_idea_to_life

from memory_manager import MemoryManager

memory_manager: Optional[MemoryManager] = None

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()


# --- Setup Main Application Logger ---
# This logger will capture general application activity, like initialization,
# high-level process steps, and overall errors.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# To ensure handlers are added only once.
if not logger.handlers:
    # 1. Console Handler: Sends logs to the standard output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Console handler added to main logger.")

    # 2. File Handler: Writes logs to a file
    LOGS_DIR = "./Logs"
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        logger.info(f"Created main logs directory: {LOGS_DIR}")
    
    # Logs will be appended to 'openfabric_activity.log'
    main_file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'openfabric_activity.log'), mode='a')
    main_file_handler.setFormatter(formatter)
    logger.addHandler(main_file_handler)
    logger.info(f"File handler added to main logger: {os.path.join(LOGS_DIR, 'openfabric_activity.log')}")

logger.info("Main application logger fully initialized.")

if not os.path.exists("OutputImage"):
        os.makedirs("OutputImage")

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    global memory_manager
    for uid, conf in configuration.items():
        logger.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf

    # Initialize MemoryManager if not already initialized
    # This ensures it's ready when the first execute() call happens
    if memory_manager is None:
        try:
            memory_manager = MemoryManager(ollama_model_name="llama3", logger_instance=logger)
            logger.info("MemoryManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryManager: {e}")



############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """
    logger.info(f"--- STARTING NEW EXECUTION ---")

    # Retrieve input
    request: InputClass = model.request
    user_prompt = request.prompt
    logger.info(f"Received user prompt: '{user_prompt}'")

    # Retrieve user config
    user_config: ConfigClass = configurations.get('super-user', None)
    logger.info(f"Current configurations state: {configurations}")

    # Initialize the Stub with app IDs
    # app_ids = user_config['app_ids'] if user_config else []
    app_ids = user_config.app_ids if user_config else []
    logger.info(f"Using app IDs: {app_ids}")
    stub = Stub(app_ids)

    # Step 0: Detect intent (remix or new creation)
    intent_type = "new_generation"
    if memory_manager is None:
        logger.warning("MemoryManager not initialized. Cannot use memory features.")
    else:
        logger.info(f"Analyzing user prompt for intent: '{user_prompt[:50]}...'")
        try:
            intent_type = detect_intent_ollama(user_prompt)
        except Exception as e:
            logger.error(f"Error detecting intent using LLM: {e}")
        logger.info(f"Detected intent: {intent_type}")

    final_prompt_for_generation = user_prompt
    recalled_creation = None

    # --- Handle Remix Intent ---
    if intent_type == "remix" and memory_manager:
        logger.info("Remix intent detected. Attempting to recall past creations.")

        # Step A: Try short-term memory first for semantic match
        logger.info(f"Searching short-term memory for: '{user_prompt}'")
        short_term_results = memory_manager.search_short_term_memory(user_prompt, n_results=1)

        logger.info(f"Searching long-term memory for semantic similarity to: '{user_prompt}'")
        long_term_results = memory_manager.search_long_term_memory(user_prompt, n_results=1)

        recalled_creation, best_score = memory_manager.select_best_memory_match(short_term_results, long_term_results)

        if recalled_creation:
            logger.info(f"Best recall from memory: '{recalled_creation['user_prompt'][:50]}...' with score {best_score}")
            recalled_expanded_prompt = recalled_creation.get('expanded_prompt', recalled_creation.get('user_prompt'))
            
            final_prompt_for_generation = (
                f"User wants to remix. Here's a detailed description of a past creation: "
                f"'{recalled_expanded_prompt}'. "
                f"Now, based on the user's new request: '{user_prompt}', "
                f"creatively combine these elements into a single, vivid, and detailed artistic description suitable for image generation. "
                f"Ensure the new description incorporates the modifications or references from the user's latest input."
            )
            logger.info(f"Constructed remix prompt for LLM expansion: '{final_prompt_for_generation}'")
        else:
            logger.info("No relevant past creation found to remix. Treating as a new creation.")
            final_prompt_for_generation = user_prompt
            intent_type = "new_generation"

    #Step 1:-  Expand prompt using local LLM
    try:
        expanded_prompt_by_llm = expand_prompt(final_prompt_for_generation)
        logger.info(f"Expanded prompt: {expanded_prompt_by_llm}")
    except Exception as e:
        logger.error(f"Error expanding prompt using LLM: {e}")
        expanded_prompt_by_llm = final_prompt_for_generation

    #Step 2:-  Bring idea to life using the expanded prompt
    try:
        image_data = bring_idea_to_life(expanded_prompt_by_llm, stub)
    except Exception as e:
        logger.error(f"Error generating image from prompt: {e}")
        image_data = None

    # Step 3: Remember Everything
    output_image_filename: Optional[str] = None

    if image_data and isinstance(image_data, bytes):
        # Save the generated image data to a file
        output_image_filename = f"./OutputImage/generated_image_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        try:
            with open(output_image_filename, 'wb') as f:
                f.write(image_data)
            logger.info(f"Generated image saved as: {output_image_filename}")

            # Step 3: Save the details to long-term and short-term memory
            saved_id = memory_manager.save_creation(
                user_prompt=user_prompt,
                expanded_prompt=expanded_prompt_by_llm
            )
            if saved_id:
                logger.info(f"Creation successfully saved to memory with ID: {saved_id}")
            else:
                logger.error("Failed to save creation to memory.")

        except Exception as e:
            logger.error(f"Failed to save image to file: {e}")
    else:
        response.message = "Failed to generate image."
        logger.error("Image generation failed or returned invalid data.")

    # Prepare response
    response: OutputClass = model.response
    response.message = f"Echo: {user_prompt}"