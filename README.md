main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, re, datetime, tempfile, fitz, docx, base64

from ppt_generator import create_ppt

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

# ---------------- CONFIG ----------------
PROJECT_ID = "drl-zenai-prod"  
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

TEXT_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL = GenerativeModel(TEXT_MODEL_NAME)



# ---------------- FASTAPI ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class ChatRequest(BaseModel):
    message: str

class ChatDocRequest(BaseModel):
    message: str
    document_text: str

class Slide(BaseModel):
    title: str
    description: str

class Section(BaseModel):
    title: str
    description: str

class Outline(BaseModel):
    title: str
    slides: List[Slide]


class EditRequest(BaseModel):
    outline: Outline
    feedback: str



class GeneratePPTRequest(BaseModel):
    description: str = ""
    outline: Optional[Outline] = None




# ---------------- HELPERS ----------------
def extract_slide_count(description: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(r"(\d+)\s*(slides?|sections?|pages?)", description, re.IGNORECASE)
    if m:
        total = int(m.group(1))
        return max(1, total - 1)
    return None if default is None else default - 1

def call_vertex(prompt: str) -> str:
    try:
        response = TEXT_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vertex AI text generation error: {e}")

def generate_title(summary: str) -> str:
    prompt = f"""Read the following summary and create a short, clear, presentation-style title.
- Keep it under 10 words
- Do not include birth dates, long sentences, or excessive details
- Just give a clean title, like a presentation heading

Summary:
{summary}
"""
    return call_vertex(prompt).strip()

def parse_points(points_text: str):
    points = []
    current_title, current_content = None, []
    lines = [re.sub(r"[#*>`]", "", ln).rstrip() for ln in points_text.splitlines()]

    for line in lines:
        if not line or "Would you like" in line:
            continue
        m = re.match(r"^\s*(Slide|Section)\s*(\d+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            if current_title:
                points.append({"title": current_title, "description": "\n".join(current_content)})
            current_title, current_content = m.group(3).strip(), []
            continue
        if line.strip().startswith("-"):
            text = line.lstrip("-").strip()
            if text:
                current_content.append(f"• {text}")
        elif line.strip().startswith(("•", "*")) or line.startswith("  "):
            text = line.lstrip("•*").strip()
            if text:
                current_content.append(f"- {text}")
        else:
            if line.strip():
                current_content.append(line.strip())

    if current_title:
        points.append({"title": current_title, "description": "\n".join(current_content)})
    return points

def extract_text(path: str, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        text_parts: List[str] = []
        doc = fitz.open(path)
        try:
            for page in doc:
                text_parts.append(page.get_text("text"))
        finally:
            doc.close()
        return "\n".join(text_parts)
    if name.endswith(".docx"):
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    if name.endswith(".txt"):
        for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def split_text(text: str, chunk_size: int = 8000, overlap: int = 300) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def generate_outline_from_desc(description: str, num_items: Optional[int], mode: str = "ppt"):
    if num_items:
        if mode == "ppt":
            prompt = f"""Create a PowerPoint outline on: {description}.
Generate exactly {num_items} content slides (⚠️ excluding the title slide).
Do NOT include a title slide — I will handle it separately.
Start from Slide 1 as the first *content slide*.
Format strictly like this:
Slide 1: <Title>
- Bullet
- Bullet
- Bullet
"""
        else:
            prompt = f"""Create a detailed Document outline on: {description}.
Generate exactly {num_items} sections (treat each section as roughly one page).
Each section should have:
- A section title
- 2–3 descriptive paragraphs (5–7 sentences each).
Do NOT use bullet points.
Format strictly like this:
Section 1: <Title>
<Paragraph 1>
<Paragraph 2>
<Paragraph 3>
"""
    else:
        if mode == "ppt":
            prompt = f"""Create a PowerPoint outline on: {description}.
Decide the most appropriate number of content slides (⚠️ excluding the title slide).
Each slide should have a short title and 3–4 bullet points.
The short title should be a single line not a double line
Do NOT include a title slide — I will handle it separately.
Format strictly like this:
Slide 1: <Title>
- Bullet
- Bullet
- Bullet
"""
        else:
            prompt = f"""Create a detailed Document outline on: {description}.
Decide the most appropriate number of sections (treat each section as roughly one page).
Each section should have:
- A section title
- 2–3 descriptive paragraphs (5–7 sentences each).
Do NOT use bullet points.
Format strictly like this:
Section 1: <Title>
<Paragraphs...>
"""
    points_text = call_vertex(prompt)
    return parse_points(points_text)

def summarize_long_text(full_text: str) -> str:
    chunks = split_text(full_text)
    if len(chunks) <= 1:
        return call_vertex(f"Summarize the following text in detail:\n\n{full_text}")
    partial_summaries = []
    for idx, ch in enumerate(chunks, start=1):
        mapped = call_vertex(f"Summarize this part of a longer document:\n\n{ch}")
        partial_summaries.append(f"Chunk {idx}:\n{mapped.strip()}")
    combined = "\n\n".join(partial_summaries)
    return call_vertex(f"Combine these summaries into one clean, well-structured summary:\n\n{combined}")

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

def clean_title(title: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", title).strip()

def save_temp_image(image_bytes, idx, title):
    output_dir = os.path.join(os.path.dirname(__file__), "generated_files", "images")
    os.makedirs(output_dir, exist_ok=True)
    safe_title = re.sub(r'[^A-Za-z0-9_.-]', '_', title)[:30]
    filename = f"{safe_title}_{idx}.png"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath

def should_generate_image(title: str, description: str) -> bool:
    """
    Decide if a slide/section really needs an image.
    Images should only be generated when a visual will
    add significant clarity (e.g., charts, diagrams, processes, comparisons).
    Avoid images for generic intro, text-heavy, or conclusion slides.
    """
    prompt = f"""
    You are deciding if an image is TRULY necessary for a presentation slide.

    Title: {title}
    Content: {description}

    Rules:
    - Say "YES" ONLY if a clear visual, diagram, chart, or illustration
      would help explain this content.
    - Say "NO" for general text slides, introductions, conclusions,
      or content that does not need a visual.
    - Avoid making every slide have an image.

    Answer strictly with YES or NO.
    """

    try:
        decision = call_vertex(prompt).strip().upper()
        return decision.startswith("Y")
    except:
        return False





# ---------------- ROUTES ----------------
@app.post("/chat")
def chat(req: ChatRequest):
    reply = call_vertex(req.message)
    return {"response": reply}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path, file.filename)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Unsupported, empty, or unreadable file content.")
    try:
        summary = summarize_long_text(text)
        title = generate_title(summary) or os.path.splitext(file.filename)[0]
        return {
            "filename": file.filename,
            "chars": len(text),
            "chunks": len(split_text(text)),
            "title": title,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

@app.post("/generate-ppt-outline")
def generate_ppt_outline(request: GeneratePPTRequest):
    title = generate_title(request.description)
    num_content_slides = extract_slide_count(request.description, default=None)
    points = generate_outline_from_desc(request.description, num_content_slides, mode="ppt")
    return {"title": title, "slides": points}

@app.post("/generate-ppt")
def generate_ppt(req: GeneratePPTRequest):
    if req.outline:
        title = clean_title(req.outline.title) or "Presentation"
        points = [{"title": clean_title(s.title), "description": s.description} for s in req.outline.slides]
    else:
        title = clean_title(generate_title(req.description))
        num_content_slides = extract_slide_count(req.description, default=None)
        points = generate_outline_from_desc(req.description, num_content_slides, mode="ppt")


    output_dir = os.path.join(os.path.dirname(__file__), "generated_files")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{sanitize_filename(title)}.pptx")

    create_ppt(title, points, filename=filename)

    return FileResponse(filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=os.path.basename(filename)
    )





@app.post("/chat-doc")
def chat_with_doc(req: ChatDocRequest):
    prompt = f"""
    You are an assistant answering based only on the provided document.
    Document:
    {req.document_text}

    Question:
    {req.message}

    Answer clearly and concisely using only the document content.
    """
    try:
        reply = call_vertex(prompt)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat-with-doc failed: {e}")



@app.get("/health")
def health():
    return {"status": "ok", "text_model": TEXT_MODEL_NAME}

@app.post("/edit-ppt-outline")
def edit_ppt_outline(req: EditRequest):
    outline_text = "\n".join(
        [f"Slide {i+1}: {s.title}\n{s.description}" for i, s in enumerate(req.outline.slides)]
    )
    prompt = f"""
    You are an assistant improving a PowerPoint outline.

    Current Outline:
    Title: {req.outline.title}
    {outline_text}

    Feedback:
    {req.feedback}

    Task:
    - Apply the feedback to refine/improve the outline.
    - Return the updated outline with the same format:
      Slide 1: <Title>
      - Bullet
      - Bullet
    - Do NOT add a title slide (I will handle it).
    """
    try:
        updated_points = parse_points(call_vertex(prompt))
        return {"title": req.outline.title, "slides": updated_points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PPT outline editing failed: {e}")

app.py
import copy
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Productivity Suite", layout="wide")
st.title("AI Productivity Suite")

# ---------------- Helpers ----------------
def extract_filename_from_cd(resp):
    cd = resp.headers.get("content-disposition", "")
    if "filename=" in cd:
        return cd.split("filename=")[-1].strip().strip('"')
    return None

def render_outline_preview(outline_data):
    if not outline_data:
        st.info("No outline available.")
        return False

    title = outline_data.get("title", "Untitled")
    slides = outline_data.get("slides", [])
    st.subheader(f"📝 Preview Outline: {title}")

    for idx, slide in enumerate(slides, start=1):
        with st.expander(f"Slide {idx}: {slide.get('title', f'Slide {idx}')}", expanded=False):
            st.markdown(slide.get("description", "").replace("\n", "\n\n"))

    return len(slides) > 0


# ---------------- STATE ----------------
defaults = {
    "messages": [],            # general chat
    "outline_chat": None,      # ppt outline
    "generated_files": [],     # past generated files
    "summary_text": None,      # uploaded doc summary
    "summary_title": None,     # uploaded doc title
    "doc_chat_history": [],    # chat with doc
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------- DISPLAY PAST CHAT ----------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

for role, content in st.session_state.doc_chat_history:
    with st.chat_message(role):
        st.markdown(content)


# ---------------- FILE UPLOAD SECTION ----------------
uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    with st.spinner("Processing uploaded file..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        try:
            res = requests.post(f"{BACKEND_URL}/upload/", files=files, timeout=180)
        except Exception as e:
            st.error(f"❌ Summarizer backend error: {e}")
            res = None

    if res and res.status_code == 200:
        data = res.json()
        st.session_state.summary_text = data.get("summary", "")
        st.session_state.summary_title = data.get("title", "Summary")
        st.success(f"✅ Document uploaded! Suggested Title: **{st.session_state.summary_title}**. You can now chat with it.")


# ---------------- ONE CHAT INPUT ----------------
if prompt := st.chat_input("💬 Type a message (general chat or ask about uploaded doc)..."):

    if st.session_state.summary_text:  
        # Doc chat mode
        st.session_state.doc_chat_history.append(("user", prompt))
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat-doc",
                json={"message": prompt, "document_text": st.session_state.summary_text},
                timeout=120,
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "⚠️ No answer")
            else:
                answer = f"❌ Error: {resp.status_code} — {resp.text}"
            st.session_state.doc_chat_history.append(("assistant", answer))
        except Exception as e:
            st.session_state.doc_chat_history.append(("assistant", f"⚠️ Backend error: {e}"))

    else:  
        # Normal chat / PPT requests
        st.session_state.messages.append(("user", prompt))
        text = prompt.lower()

        try:
            if "ppt" in text or "presentation" in text or "slides" in text:
                with st.spinner("Generating PPT outline..."):
                    resp = requests.post(f"{BACKEND_URL}/generate-ppt-outline", json={"description": prompt}, timeout=120)
                    if resp.status_code == 200:
                        st.session_state.outline_chat = resp.json()
                        st.session_state.messages.append(("assistant", "✅ PPT outline generated! Preview below."))
                    else:
                        st.session_state.messages.append(("assistant", f"❌ PPT outline failed: {resp.text}"))
            else:
                resp = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt}, timeout=60)
                bot_reply = resp.json().get("response", "⚠️ Error")
                st.session_state.messages.append(("assistant", bot_reply))
        except Exception as e:
            st.session_state.messages.append(("assistant", f"⚠️ Backend error: {e}"))

    st.rerun()


# ---------------- OUTLINE PREVIEW + ACTIONS ----------------
if st.session_state.outline_chat:
    outline = st.session_state.outline_chat
    render_outline_preview(outline)

    new_title = st.text_input("📌 Edit Title", value=outline.get("title", "Untitled"))
    feedback_box = st.text_area("✏️ Feedback for outline (optional):", value="")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback"):
            with st.spinner("Updating outline..."):
                try:
                    edit_payload = {"outline": outline, "feedback": feedback_box}
                    resp = requests.post(f"{BACKEND_URL}/edit-ppt-outline", json=edit_payload, timeout=120)
                    if resp.status_code == 200:
                        updated_outline = resp.json()
                        updated_outline["title"] = new_title.strip() if new_title else updated_outline["title"]
                        st.session_state.outline_chat = updated_outline
                        st.success("✅ Outline updated!")
                        st.rerun()
                    else:
                        st.error(f"❌ Edit failed: {resp.status_code} — {resp.text}")
                except Exception as e:
                    st.error(f"❌ Edit error: {e}")

    with col2:
        if st.button("✅ Generate PPT"):
            with st.spinner("Generating PPT..."):
                try:
                    outline_to_send = copy.deepcopy(outline)
                    outline_to_send["title"] = new_title.strip() if new_title else outline_to_send["title"]

                    resp = requests.post(f"{BACKEND_URL}/generate-ppt", json={"outline": outline_to_send}, timeout=180)
                    if resp.status_code == 200:
                        filename = extract_filename_from_cd(resp) or "presentation.pptx"
                        st.success("✅ PPT generated successfully!")
                        st.download_button(
                            "⬇️ Download PPT",
                            data=resp.content,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                        st.session_state.generated_files.append({
                            "type": "ppt",
                            "filename": filename,
                            "content": resp.content,
                        })
                        st.session_state.outline_chat = None
                    else:
                        st.error(f"❌ PPT generation failed: {resp.status_code} — {resp.text}")
                except Exception as e:
                    st.error(f"❌ PPT generation error: {e}")




