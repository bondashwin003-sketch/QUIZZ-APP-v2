import os
import json
import fitz  # PyMuPDF (replaces pdf2image, no Poppler needed!)
import streamlit as st
from typing import List, Literal
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# =====================================================================
# 1. Define Structured Output Schema using Pydantic
# =====================================================================

class MCQItem(BaseModel):
    question: str = Field(description="The multiple-choice question stem.")
    options: List[str] = Field(
        description="List of 4 options labeled A, B, C, and D."
    )
    correct_answer: str = Field(
        description="The exact correct option letter (e.g., 'A', 'B', 'C', or 'D')."
    )
    explanation: str = Field(
        description="Detailed explanation of why the answer is correct and why distractors are wrong."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        description="Difficulty level: Easy (recall), Medium (application), Hard (analytical/synthesis)."
    )
    topic_tag: str = Field(
        description="The specific sub-topic or concept tested by this question."
    )

class DocumentExtractionAndQuiz(BaseModel):
    extracted_text_summary: str = Field(
        description="Complete and accurate transcription/summary of the text, equations, and diagrams detected in the document."
    )
    mcqs: List[MCQItem] = Field(
        description="List of generated Multiple Choice Questions across difficulty levels."
    )

# =====================================================================
# 2. Convert Uploaded PDF / Image into PIL Images in Memory
# =====================================================================

def process_uploaded_file(uploaded_file, start_page: int = 1, end_page: int = 5) -> List[Image.Image]:
    """Reads an uploaded Streamlit file (PDF or Image) directly from memory."""
    images = []
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if file_name.endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        actual_end = min(end_page, total_pages)
        
        for page_num in range(start_page - 1, actual_end):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    else:
        img = Image.open(uploaded_file)
        images.append(img)

    return images

# =====================================================================
# 3. Call Gemini 3.8 Flash to Extract Text & Generate MCQs
# =====================================================================

def generate_quiz_from_images(
    api_key: str,
    images: List[Image.Image],
    num_easy: int,
    num_medium: int,
    num_hard: int
) -> DocumentExtractionAndQuiz:
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert OCR system and academic examiner.
    Carefully examine the attached document page(s), which may contain typed textbook text, 
    scanned pages, tables, diagrams, or handwritten notes.

    TASK 1: CONTENT EXTRACTION
    Accurately read and transcribe the core text, formulas, definitions, and key points 
    from the provided page(s) into the `extracted_text_summary` field.

    TASK 2: TIERED MCQ GENERATION
    Based strictly on the extracted content, generate a total of {num_easy + num_medium + num_hard} 
    high-quality Multiple Choice Questions (MCQs) distributed across three difficulty levels:
    
    - {num_easy} EASY questions: Direct factual recall, definitions, and basic concepts.
    - {num_medium} MEDIUM questions: Conceptual understanding, cause-and-effect, and practical application.
    - {num_hard} HARD questions: Multi-step analytical reasoning, numerical problem-solving, or assertion-reason logic.

    Format each option starting with its letter, like "A) ...", "B) ...", "C) ...", "D) ...".
    Ensure `correct_answer` is just the single letter ("A", "B", "C", or "D").
    """

    contents = images + [prompt]

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DocumentExtractionAndQuiz,
            temperature=0.3,
        ),
    )

    result_data = json.loads(response.text)
    return DocumentExtractionAndQuiz(**result_data)

# =====================================================================
# 4. Streamlit Web UI
# =====================================================================

st.set_page_config(page_title="AI Textbook & Note Quiz Generator", page_icon="📚", layout="wide")
st.title("📚 AI Textbook, PDF & Handwritten Note MCQ Generator")
st.write("Upload any scanned textbook PDF, digital document, or handwritten note image to extract content and generate difficulty-tiered MCQs.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check Streamlit Secrets or Environment first, otherwise ask user in UI
    default_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    api_key = st.text_input("Gemini API Key", value=default_key, type="password", help="Get your key from aistudio.google.com/apikey")
    
    st.subheader("Difficulty Distribution")
    num_easy = st.slider("Easy Questions", 0, 10, 3)
    num_medium = st.slider("Medium Questions", 0, 10, 3)
    num_hard = st.slider("Hard Questions", 0, 10, 3)
    
    st.subheader("PDF Page Range")
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("Start Page", min_value=1, value=1)
    with col2:
        end_page = st.number_input("End Page", min_value=1, value=3)

# File Uploader
uploaded_file = st.file_uploader(
    "Upload a PDF or Image (.pdf, .png, .jpg, .jpeg, .webp)", 
    type=["pdf", "png", "jpg", "jpeg", "webp"]
)

if uploaded_file and st.button("🚀 Extract Content & Generate MCQs", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar first!")
    elif end_page < start_page:
        st.error("End Page must be greater than or equal to Start Page.")
    else:
        with st.spinner("Rendering pages and analyzing content with Gemini Vision..."):
            try:
                images = process_uploaded_file(uploaded_file, start_page, end_page)
                st.info(f"Processed {len(images)} page(s). Generating questions...")
                
                quiz_data = generate_quiz_from_images(
                    api_key=api_key,
                    images=images,
                    num_easy=num_easy,
                    num_medium=num_medium,
                    num_hard=num_hard
                )
                st.session_state["quiz_data"] = quiz_data
                st.success("Quiz generated successfully!")
            except Exception as e:
                st.error(f"Error processing file: {e}")

# Display Results if available in session state
if "quiz_data" in st.session_state:
    data: DocumentExtractionAndQuiz = st.session_state["quiz_data"]
    
    tab1, tab2, tab3 = st.tabs(["📝 Generated MCQs", "📖 Extracted Notes Summary", "⬇️ Download"])
    
    with tab1:
        for difficulty in ["Easy", "Medium", "Hard"]:
            tier_qs = [q for q in data.mcqs if q.difficulty == difficulty]
            if tier_qs:
                st.subheader(f"🔥 {difficulty} Level ({len(tier_qs)} Questions)")
                for idx, q in enumerate(tier_qs, 1):
                    with st.container(border=True):
                        st.markdown(f"**Q{idx}. [{q.topic_tag}]** {q.question}")
                        for opt in q.options:
                            st.write(opt)
                        with st.expander("Show Answer & Explanation"):
                            st.markdown(f"**Correct Answer:** `{q.correct_answer}`")
                            st.markdown(f"**Explanation:** {q.explanation}")
                            
    with tab2:
        st.subheader("Extracted Text & Formulas")
        st.write(data.extracted_text_summary)
        
    with tab3:
        st.download_button(
            label="📥 Download Quiz as JSON",
            data=data.model_dump_json(indent=2),
            file_name="generated_mcqs.json",
            mime="application/json"
        )
