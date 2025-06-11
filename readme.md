# 🧠 Prompt-to-Image AI Creator

Turn your ideas into real pictures. This app lets you write a simple idea and see it change into an image — using local AI, strong visual tools, and a memory that always remembers your work.

---

## ✨ What It Does

Type something like:

> “A vibrant coral reef with colorful exotic fish swimming around.”

> "Build a battle-worn medieval knight riding a horse in armor."

And here's what happens behind the scenes:

1. Your idea gets refined by a locally running LLM (via Ollama).
2. That expanded prompt is sent to a Text-to-Image app to create a visual.
3. Every prompt and output is remembered using smart memory — ready to recall later.

You can even say:

> “The same medieval knight now holding a glowing sword and shield.”

And the system remembers exactly what you meant.

---

## 📁 Files You Should Know

- `ignite.py`: The starting point for everything.
- `main.py`: The starting point for everything.
- `llm.py`: Talks to the local LLM and gets creative with your prompt.
- `text2image.py`: Handles the chain from text to image.
- `memory_manager.py`: Stores and retrieves both short-term and long-term memory.
- `start.sh`: Run the app.
- `Dockerfile`: Container support, if you prefer Docker.
- `app.py`: Streamlit UI (Additional UI file)

---

## Getting Started

Follow these steps to set up and run the application.

## Prerequisites

* Python 3.10 or later
* Poetry

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd ai-test
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
    This will create a virtual environment and install all necessary packages.

## Running the Application

### ⚙️ Two Ways to Use It

1. **Swagger UI (API Playground)** 
  
    To run the main application and access its API through Swagger, use Poetry to execute start.sh within the project's virtual environment:

    ```bash
    poetry run bash start.sh
    ```

    Upon successful startup (you'll likely see Flask server messages), visit the Swagger UI in your browser:
    http://localhost:8888/swagger-ui/#/App/post_execution

    (Alternatively, you can run using `poetry run python ignite.py` or using Docker)

2. **Streamlit Interface (Optional)**  
   A simple Streamlit app is available in `app.py` for a more interactive, visual experience.  
   Just run:
   ```bash
   poetry run streamlit run app.py
   ```

## 🧠 Memory

- Short-term: Works inside a single run/session.
- Long-term: Saved across runs using a ChromaDB.

Perfect for building on past ideas or creating variations.

---

## 🧪 Tech Behind It

- Python + Openfabric SDK
- Ollama (for local LLM)
- One Openfabric apps:
  - Text-to-Image: `f0997a01-d6d3-a5fe-53d8-561300318557`
- Streamlit for UI

## 🖼 Bonus Features

- Clear logs for each step
- Persistent memory system
- Extensible for new creative workflows

