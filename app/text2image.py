import logging
def bring_idea_to_life(prompt: str, stub) -> str:
    """
    Takes a vivid prompt and generates a 3D model by chaining:
    - Text-to-Image
    - Image-to-3D

    Args:
        prompt (str): The vivid visual description from the LLM
        stub (Stub): The stub object initialized with authorized node access

    Returns:
        byte: 2d Image
    """

    # ---- Step 1: Call Text-to-Image App ----
    app_ids = list(stub._connections.keys())

    logging.info(f"Available app IDs: {app_ids}")

    text_to_image_app_id = app_ids[0]
    #image_to_3d_app_id = app_ids[1]
    
    text_to_image_input = {
        "prompt": prompt
    }

    text_to_image_output = stub.call(text_to_image_app_id, text_to_image_input).get("result")
    return text_to_image_output

    # Step 2: Convert image to 3D model
    # app_id not working of 3d model (502 error)
