from ollama import Client

client = Client(host='http://localhost:11434')

def expand_prompt(prompt: str, model: str = 'llama3') -> str:
    system_prompt = (
        "You are a visual designer assistant. Expand the user's prompt into a detailed, vivid visual description "
        "with rich textures, moods, lighting, and composition that can guide image generation."
    )

    response = client.chat(model=model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ])

    return response['message']['content'].strip()

def detect_intent_ollama(user_input: str) -> str:
    prompt = f"""
        You are an assistant that classifies user requests about image generation into exactly one of two categories:

        - "new_generation": The user wants a completely new and original image prompt, with no reference or relation to previous images.

        - "remix": The user wants a variation, modification, or remix based on a previous or existing image prompt. This includes explicit or implicit references to past images, such as mentioning "previous," "earlier," "another version," "like before," or any hint that connects the request to prior creations.

        Based ONLY on the user's input below, reply with exactly one word: "new_generation" or "remix".  
        Do NOT provide any explanations or additional text.

        User input: "{user_input}"
    """


    # Create an Ollama client and call the llama3 model
    response = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    intent = response['message']['content'].strip().lower()
    if intent not in ["remix", "new_generation"]:
        intent = "new_generation"  # fallback default

    return intent

