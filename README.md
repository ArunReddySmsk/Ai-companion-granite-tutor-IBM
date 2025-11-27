# AI Campus Companion – Granite Tutor

A Streamlit-based educational assistant powered by **Ollama** and **Granite 3.1 MoE**.

## Features
- **🐍 Python Tutor**: Personalized coding help and practice questions.
- **🏫 Campus Admin Assistant**: Answers questions about fees, exams, and rules using local documents (RAG).
- **🧭 Course & Career Guide**: Recommends electives and certifications based on your interests.

---

## 🚀 Setup Instructions

### 1. Install Ollama
This app requires **Ollama** to run the AI model locally.

1.  Go to [ollama.com](https://ollama.com).
2.  Click **Download** and choose **Windows**.
3.  Run the installer and follow the setup instructions.
4.  Once installed, open a terminal (Command Prompt or PowerShell) and type `ollama` to verify it's working.

### 2. Download the Model
You need the `granite3.1-moe:latest` model.

1.  Open your terminal.
2.  Run the following command:
    ```bash
    ollama pull granite3.1-moe:latest
    ```
3.  Wait for the download to complete (it may take a few minutes depending on your internet speed).

### 3. Install Python Dependencies
Make sure you have Python installed. Then, in the project folder:

```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit app:

```bash
streamlit run app.py
```

The app should open automatically in your browser at `http://localhost:8501`.

---

## 📂 Project Structure
- `app.py`: Main application code.
- `data/campus_docs/`: Place your PDF or Text documents here for the Admin Assistant to read.
- `requirements.txt`: List of Python libraries.

## ⚠️ Note
- Ensure Ollama is running in the background (it usually starts automatically after installation).
- If the Admin Assistant says "No relevant documents found", make sure you have files in `data/campus_docs/`.
