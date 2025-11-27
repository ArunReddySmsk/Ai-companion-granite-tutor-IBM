import os
import json
from typing import List, Dict, Tuple

import streamlit as st
import requests
import pandas as pd
import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.embeddings import Embeddings

# -----------------------------------------------------------------------------
# HOW TO CONFIGURE (.env in project root)
# -----------------------------------------------------------------------------
# IBM_WATSONX_API_KEY=your_ibm_cloud_api_key_here
# IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
# IBM_WATSONX_PROJECT_ID=your_watsonx_project_id_here
# IBM_WATSONX_CHAT_MODEL=ibm/granite-3-8b-instruct
# IBM_WATSONX_EMBED_MODEL=ibm-all-minilm-l6-v2-1024
# -----------------------------------------------------------------------------

load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
IBM_WX_API_KEY = os.getenv("IBM_WATSONX_API_KEY", "")
IBM_WX_URL = os.getenv("IBM_WATSONX_URL", "")
IBM_WX_PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID", "")

IBM_WX_CHAT_MODEL_ID = os.getenv("IBM_WATSONX_CHAT_MODEL", "ibm/granite-3-8b-instruct")
IBM_WX_EMBED_MODEL_ID = os.getenv("IBM_WATSONX_EMBED_MODEL", "ibm-all-minilm-l6-v2-1024")

DATA_DIR = os.path.join("data", "campus_docs")

if not (IBM_WX_API_KEY and IBM_WX_URL and IBM_WX_PROJECT_ID):
    st.sidebar.warning("IBM watsonx.ai credentials are missing. Set them in .env before using the app.")


# -----------------------------------------------------------------------------
# IBM WATSONX.AI HELPERS (CORRECT CREDENTIALS USAGE)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_watsonx_credentials() -> Credentials:
    """
    Shared Credentials object for watsonx.ai. [web:84][web:90]
    """
    return Credentials(
        url=IBM_WX_URL,
        api_key=IBM_WX_API_KEY,
    )


@st.cache_resource
def get_watsonx_chat_model() -> ModelInference:
    """
    Text/chat model using ModelInference. [web:82]
    """
    creds = get_watsonx_credentials()
    params = {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "decoding_method": "greedy",
        "top_p": 1.0,
        "top_k": 50,
    }
    model = ModelInference(
        model_id=IBM_WX_CHAT_MODEL_ID,
        params=params,
        credentials=creds,
        project_id=IBM_WX_PROJECT_ID,
    )
    return model


@st.cache_resource
def get_watsonx_embed_model() -> Embeddings:
    """
    Embedding model for RAG. [web:41]
    """
    creds = get_watsonx_credentials()
    embed_model = Embeddings(
        model_id=IBM_WX_EMBED_MODEL_ID,
        credentials=creds,
        project_id=IBM_WX_PROJECT_ID,
    )
    return embed_model


def get_watsonx_embedding(text: str) -> List[float]:
    """
    Generate embeddings using IBM watsonx.ai Embeddings API. [web:41]
    """
    if not text.strip():
        return []
    try:
        embed_model = get_watsonx_embed_model()
        # embed_query returns list[float] for a single string. [web:41]
        vector = embed_model.embed_query(text)
        return vector or []
    except Exception as e:
        st.error(f"Error getting embedding from watsonx.ai: {e}")
        return []


def call_watsonx_chat(
    system_prompt: str,
    user_message: str,
    context_text: str = "",
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """
    Call IBM watsonx.ai chat/text generation with Granite via ModelInference.chat. [web:82]
    """
    try:
        chat_model = get_watsonx_chat_model()

        full_user_content = user_message
        if context_text:
            full_user_content = f"CONTEXT:\n{context_text}\n\nQUESTION / TASK:\n{user_message}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_content},
        ]

        # Override per-call params if needed
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

        # Use chat() to keep role structure. [web:82]
        resp = chat_model.chat(messages=messages, params=params)

        # Typical shape: resp["choices"][0]["message"]["content"]. [web:82]
        choices = resp.get("choices", [])
        if choices and isinstance(choices, list):
            msg = choices[0].get("message", {})
            text = msg.get("content", "")
            return text.strip() or "No response from model."
        # Fallback to generate_text() style
        text = resp.get("generated_text") or ""
        return text.strip() or "No response from model."
    except Exception as e:
        return f"Error calling watsonx.ai: {e}"


def generate_llm_response(
    system_prompt: str,
    user_message: str,
    context_text: str = "",
    max_tokens: int = 512,
) -> str:
    return call_watsonx_chat(system_prompt, user_message, context_text, max_tokens=max_tokens)


# -----------------------------------------------------------------------------
# RAG HELPER FUNCTIONS (Campus Admin Assistant)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_rag_index():
    """
    Loads documents from data/campus_docs, chunks them, embeds them using watsonx.ai,
    and builds a FAISS index. Returns (index, chunks).
    """
    docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf"))]

    if not files:
        return None, []

    for f in files:
        path = os.path.join(DATA_DIR, f)
        text = ""
        if f.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
        elif f.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    text += extracted + "\n"
            except Exception as e:
                print(f"Error reading PDF {f}: {e}")
                continue

        if text:
            docs.append({"filename": f, "text": text})

    # Chunking
    chunks = []
    CHUNK_SIZE = 1000
    OVERLAP = 100

    for doc in docs:
        text = doc["text"]
        filename = doc["filename"]
        for i in range(0, len(text), CHUNK_SIZE - OVERLAP):
            chunk_text = text[i : i + CHUNK_SIZE]
            chunks.append({"text": chunk_text, "filename": filename})

    if not chunks:
        return None, []

    # Embedding & Indexing
    embeddings = []
    valid_chunks = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, chunk in enumerate(chunks):
        status_text.text(f"Embedding chunk {i+1}/{len(chunks)}...")
        emb = get_watsonx_embedding(chunk["text"])
        if emb:
            embeddings.append(emb)
            valid_chunks.append(chunk)
        progress_bar.progress((i + 1) / len(chunks))

    status_text.empty()
    progress_bar.empty()

    if not embeddings:
        return None, []

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    return index, valid_chunks


def rag_query(question: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
    index, chunks = load_rag_index()

    if index is None or not chunks:
        return "", []

    query_embedding = get_watsonx_embedding(question)
    if not query_embedding:
        return "", []

    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)

    retrieved_chunks = []
    context_text = ""

    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            retrieved_chunks.append(chunk)
            context_text += f"--- Source: {chunk['filename']} ---\n{chunk['text']}\n\n"

    return context_text, retrieved_chunks


# -----------------------------------------------------------------------------
# COURSE DATA (Feature 3)
# -----------------------------------------------------------------------------
COURSES_DB = [
    {"code": "CS101", "name": "Intro to Python", "tags": ["AI", "Web Dev", "Data Science"], "semester": 1, "desc": "Basics of Python programming."},
    {"code": "CS201", "name": "Data Structures", "tags": ["AI", "Web Dev", "Cloud"], "semester": 3, "desc": "Core data structures and algorithms."},
    {"code": "AI301", "name": "Artificial Intelligence", "tags": ["AI", "Data Science"], "semester": 5, "desc": "Intro to AI agents and search."},
    {"code": "ML302", "name": "Machine Learning", "tags": ["AI", "Data Science"], "semester": 5, "desc": "Supervised and unsupervised learning."},
    {"code": "WEB401", "name": "Full Stack Dev", "tags": ["Web Dev"], "semester": 7, "desc": "MERN stack development."},
    {"code": "CLD402", "name": "Cloud Computing", "tags": ["Cloud", "DevOps"], "semester": 7, "desc": "AWS and Azure fundamentals."},
    {"code": "SEC403", "name": "Cybersecurity Basics", "tags": ["Cybersecurity"], "semester": 5, "desc": "Network security and cryptography."},
    {"code": "DS404", "name": "Big Data Analytics", "tags": ["Data Science"], "semester": 7, "desc": "Hadoop and Spark."},
]

CERTIFICATIONS_DB = {
    "AI": ["TensorFlow Developer", "AWS Machine Learning Specialty"],
    "Web Dev": ["Meta Front-End Developer", "AWS Certified Developer"],
    "Data Science": ["Google Data Analytics", "IBM Data Science Professional"],
    "Cloud": ["AWS Solutions Architect", "Azure Fundamentals"],
    "Cybersecurity": ["CompTIA Security+", "CEH (Certified Ethical Hacker)"],
    "DevOps": ["Docker Certified Associate", "Kubernetes Administrator"],
}


# -----------------------------------------------------------------------------
# UI RENDER FUNCTIONS
# -----------------------------------------------------------------------------
def render_header():
    st.title("🎓 AI Campus Companion – Granite Tutor (IBM watsonx.ai)")
    st.markdown(
        """
    <style>
    .info-box {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    <div class="info-box">
        <strong>Prototype Assistant:</strong> This tool uses IBM Granite models via <code>watsonx.ai</code> to help with studies and campus info.
        <br><em>Disclaimer: Not an official university system. Always verify critical info.</em>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_python_tutor_tab():
    st.header("🐍 Personalized Python Tutor")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        level = st.selectbox("Your Level", ["Beginner", "Intermediate", "Advanced"])
        topic = st.selectbox(
            "Topic",
            ["Variables", "Data Types", "If-Else", "Loops", "Functions", "Lists", "Dictionaries", "OOP Basics", "Error Handling"],
        )

        user_doubt = st.text_area(
            "Specific Doubt or Code Snippet (Optional)",
            placeholder="e.g., I don't understand how while loops work...",
        )

        generate_questions = st.checkbox("Generate Practice Questions")
        num_questions = 3
        include_solutions = False

        if generate_questions:
            num_questions = st.number_input("Number of Questions", 1, 5, 3)
            include_solutions = st.checkbox("Include Step-by-Step Solutions")

        if st.button("Start Tutor Session"):
            with st.spinner("Granite is thinking..."):
                system_prompt = (
                    "You are a friendly and patient Python tutor. Explain concepts clearly with analogies and simple examples. "
                    "If the user asks for practice questions, provide them. "
                    "If solutions are requested, provide them at the end separated by '--- Solutions ---'."
                )

                user_req = f"Level: {level}\nTopic: {topic}\n"
                if user_doubt:
                    user_req += f"Specific Doubt: {user_doubt}\n"

                if generate_questions:
                    user_req += f"Please generate {num_questions} practice questions (Difficulty: {level})."
                    if include_solutions:
                        user_req += " Include detailed solutions at the end."
                else:
                    user_req += "Just explain the concept with examples."

                response = generate_llm_response(system_prompt, user_req)
                st.session_state["tutor_response"] = response

    with col2:
        if "tutor_response" in st.session_state:
            st.markdown("### 💡 Explanation & Practice")

            content = st.session_state["tutor_response"]
            if "--- Solutions ---" in content:
                parts = content.split("--- Solutions ---")
                main_content = parts[0]
                solutions = parts[1] if len(parts) > 1 else ""

                st.markdown(main_content)
                if solutions:
                    with st.expander("📝 View Solutions"):
                        st.markdown(solutions)
            else:
                st.markdown(content)


def render_admin_assistant_tab():
    st.header("🏫 Campus Admin Assistant")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("Ask about fees, hostels, exams, or library rules.")
        topic = st.selectbox("Topic Filter", ["General", "Fees", "Exams", "Hostel", "Library"])
        question = st.text_area("Your Question", height=100)

        if st.button("Get Answer"):
            if not question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching campus docs..."):
                    context, sources = rag_query(question)

                    if not context:
                        st.error("No relevant documents found in `data/campus_docs`. Please add PDF/TXT files there.")
                    else:
                        system_prompt = (
                            "You are a campus administrative assistant. Answer ONLY from the provided CONTEXT. "
                            "If the answer is not present, say you are not sure and suggest contacting the admin office. "
                            "Do not invent policies or numbers."
                        )

                        answer = generate_llm_response(system_prompt, question, context_text=context)
                        st.session_state["admin_answer"] = answer
                        st.session_state["admin_sources"] = sources

    with col2:
        if "admin_answer" in st.session_state:
            st.markdown("### 📋 Answer")
            st.info(st.session_state["admin_answer"])

            if "admin_sources" in st.session_state and st.session_state["admin_sources"]:
                with st.expander("🔍 Sources / Matched Sections"):
                    for src in st.session_state["admin_sources"]:
                        st.markdown(f"**File:** `{src['filename']}`")
                        st.text(src["text"][:300] + "...")
                        st.divider()

            st.caption("⚠️ This assistant may not reflect the latest official policies. Always verify with the university website.")


def render_course_counselor_tab():
    st.header("🧭 Course Finder & Career Counselor")

    col1, col2 = st.columns([1, 2])

    with col1:
        target_career = st.text_input("Target Career", placeholder="e.g., Data Scientist, Cloud Engineer")
        interests = st.multiselect("Interests", ["AI", "Web Dev", "Data Science", "Cloud", "Cybersecurity", "DevOps"])
        semester = st.selectbox("Current Semester", [1, 2, 3, 4, 5, 6, 7, 8])

        if st.button("Find Courses & Advice"):
            if not target_career and not interests:
                st.warning("Please specify a career goal or interests.")
            else:
                with st.spinner("Analyzing curriculum..."):
                    relevant_courses = [
                        c
                        for c in COURSES_DB
                        if (not interests or any(tag in interests for tag in c["tags"]))
                        and c["semester"] >= semester
                    ]

                    relevant_certs = []
                    for i in interests:
                        if i in CERTIFICATIONS_DB:
                            relevant_certs.extend(CERTIFICATIONS_DB[i])

                    system_prompt = (
                        "You are an academic and career counselor. "
                        "Analyze the student's profile and suggested courses. "
                        "Explain WHY these courses are good for their career. "
                        "Suggest additional skills or certifications not listed if appropriate. "
                        "Warn that official curriculum constraints apply."
                    )

                    context = f"Target Career: {target_career}\nInterests: {interests}\nSemester: {semester}\n"
                    context += f"Available Electives Matching Interests: {json.dumps(relevant_courses)}\n"
                    context += f"Suggested Certifications: {json.dumps(relevant_certs)}"

                    advice = generate_llm_response(
                        system_prompt,
                        "Provide career guidance and course selection advice.",
                        context_text=context,
                    )

                    st.session_state["counselor_courses"] = relevant_courses
                    st.session_state["counselor_certs"] = relevant_certs
                    st.session_state["counselor_advice"] = advice

    with col2:
        if "counselor_advice" in st.session_state:
            st.markdown("### 🎓 Recommended Electives")
            if st.session_state["counselor_courses"]:
                df = pd.DataFrame(st.session_state["counselor_courses"])
                st.dataframe(df[["code", "name", "semester", "desc"]], hide_index=True)
            else:
                st.write("No specific electives found matching filters for your semester.")

            st.markdown("### 🏆 Suggested Certifications")
            if st.session_state["counselor_certs"]:
                for cert in set(st.session_state["counselor_certs"]):
                    st.markdown(f"- {cert}")
            else:
                st.write("No specific certifications mapped to these interests.")

            st.markdown("### 🤖 AI Career Guidance")
            st.write(st.session_state["counselor_advice"])


def render_about_tab():
    st.header("ℹ️ About & Limitations")
    st.markdown(
        """
    ### What is AI Campus Companion?
    This is a **student-built prototype** designed to assist with:
    - 🐍 **Python Learning**: Explanations and practice.
    - 🏫 **Campus Queries**: Answering admin questions using local documents.
    - 🧭 **Career Guidance**: Suggesting courses and paths.

    ### 🧠 Technology
    - **LLM**: IBM Granite model (for example, `ibm/granite-3-8b-instruct`) via IBM watsonx.ai APIs.
    - **RAG**: Uses IBM embeddings with FAISS to search `data/campus_docs`. [web:38][web:56]
    - **Framework**: Built with Streamlit.

    ### ⚠️ Limitations & Disclaimers
    - **Not Official**: This is NOT an official university system.
    - **Accuracy**: The AI may hallucinate or provide outdated info.
    - **Verification**: Always verify fees, dates, and rules with the admin office.
    - **Data**: The admin assistant only knows what is in the `data/campus_docs` folder.
    """
    )


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI Campus Companion", page_icon="🎓", layout="wide")

    render_header()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🐍 Python Tutor",
            "🏫 Campus Admin Assistant",
            "🧭 Course & Career Guide",
            "ℹ️ About & Limitations",
        ]
    )

    with tab1:
        render_python_tutor_tab()

    with tab2:
        render_admin_assistant_tab()

    with tab3:
        render_course_counselor_tab()

    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
