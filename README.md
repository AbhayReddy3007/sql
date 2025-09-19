main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, re, tempfile, fitz, docx, requests

from ppt_generator import create_ppt

# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

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
def call_gemini(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

def extract_text(path: str, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        text_parts = []
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
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def split_text(text: str, chunk_size: int = 8000, overlap: int = 300):
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def summarize_long_text(full_text: str) -> str:
    chunks = split_text(full_text)
    if len(chunks) <= 1:
        return call_gemini(f"Summarize the following text in detail:\n\n{full_text}")
    partial_summaries = []
    for idx, ch in enumerate(chunks, start=1):
        mapped = call_gemini(f"Summarize this part of a longer document:\n\n{ch}")
        partial_summaries.append(f"Chunk {idx}:\n{mapped.strip()}")
    combined = "\n\n".join(partial_summaries)
    return call_gemini(f"Combine these summaries into one clean, well-structured summary:\n\n{combined}")

def extract_slide_count(description: str, default: Optional[int] = None) -> Optional[int]:
    m = re.search(r"(\d+)\s*(slides?|sections?|pages?)", description, re.IGNORECASE)
    if m:
        total = int(m.group(1))
        return max(1, total - 1)
    return None if default is None else default - 1

def generate_title(summary: str) -> str:
    prompt = f"Create a short presentation title (under 10 words) for: {summary}"
    return call_gemini(prompt)

def parse_points(points_text: str):
    points = []
    current_title, current_content = None, []
    lines = [re.sub(r"[#*>`]", "", ln).rstrip() for ln in points_text.splitlines()]

    for line in lines:
        if not line: continue
        m = re.match(r"^\s*Slide\s*(\d+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            if current_title:
                points.append({"title": current_title, "description": "\n".join(current_content)})
            current_title, current_content = m.group(2).strip(), []
            continue
        if line.strip().startswith("-"):
            current_content.append("• " + line.lstrip("-").strip())
        else:
            current_content.append(line.strip())
    if current_title:
        points.append({"title": current_title, "description": "\n".join(current_content)})
    return points

def generate_outline_from_desc(description: str, num_items: Optional[int]):
    if num_items:
        prompt = f"""Create a PowerPoint outline on: {description}.
Generate exactly {num_items} content slides (excluding the title slide).
Format strictly like:
Slide 1: <Title>
- Bullet
- Bullet
"""
    else:
        prompt = f"""Create a PowerPoint outline on: {description}.
Pick the most appropriate number of content slides (excluding the title slide).
Format strictly like:
Slide 1: <Title>
- Bullet
- Bullet
"""
    points_text = call_gemini(prompt)
    return parse_points(points_text)


# ---------------- ROUTES ----------------
@app.post("/chat")
def chat(req: ChatRequest):
    reply = call_gemini(req.message)
    return {"response": reply}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path, file.filename)
    finally:
        os.remove(tmp_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Unsupported or empty file.")
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
        reply = call_gemini(prompt)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat-with-doc failed: {e}")

@app.post("/generate-ppt-outline")
def generate_ppt_outline(request: GeneratePPTRequest):
    title = generate_title(request.description)
    num_content_slides = extract_slide_count(request.description, default=None)
    points = generate_outline_from_desc(request.description, num_content_slides)
    return {"title": title, "slides": points}

@app.post("/generate-ppt")
def generate_ppt(req: GeneratePPTRequest):
    if req.outline:
        title = req.outline.title or "Presentation"
        points = [{"title": s.title, "description": s.description} for s in req.outline.slides]
    else:
        title = generate_title(req.description)
        num_content_slides = extract_slide_count(req.description, default=None)
        points = generate_outline_from_desc(req.description, num_content_slides)

    output_dir = os.path.join(os.path.dirname(__file__), "generated_files")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', title)}.pptx")

    create_ppt(title, points, filename=filename)

    return FileResponse(filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=os.path.basename(filename)
    )

@app.post("/edit-ppt-outline")
def edit_ppt_outline(req: EditRequest):
    outline_text = "\n".join(
        [f"Slide {i+1}: {s.title}\n{s.description}" for i, s in enumerate(req.outline.slides)]
    )
    prompt = f"""
    Improve this PPT outline based on feedback.

    Current Outline:
    Title: {req.outline.title}
    {outline_text}

    Feedback:
    {req.feedback}

    Return strictly in this format:
    Slide 1: <Title>
    - Bullet
    - Bullet
    """
    updated_points = parse_points(call_gemini(prompt))
    return {"title": req.outline.title, "slides": updated_points}

@app.get("/health")
def health():
    return {"status": "ok", "text_model": "gemini-2.0-flash"}

app.py
import copy
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI PPT Generator", layout="wide")
st.title("💡 Chatbot (PPT Generator)")

# ---------------- STATE ----------------
if "messages" not in st.session_state: st.session_state.messages = []
if "outline_chat" not in st.session_state: st.session_state.outline_chat = None
if "generated_files" not in st.session_state: st.session_state.generated_files = []
if "summary_text" not in st.session_state: st.session_state.summary_text = None
if "summary_title" not in st.session_state: st.session_state.summary_title = None
if "doc_chat_history" not in st.session_state: st.session_state.doc_chat_history = []
if "outline_from_summary" not in st.session_state: st.session_state.outline_from_summary = None


# ---------------- CHAT HISTORY ----------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# ---------------- PAST GENERATED FILES ----------------
for i, file_info in enumerate(st.session_state.generated_files):
    with st.chat_message("assistant"):
        st.markdown("✅ PPT generated earlier!")
        st.download_button(
            "⬇️ Download PPT",
            data=file_info["content"],
            file_name=file_info["filename"],
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            key=f"past_download_ppt_{i}"
        )

# ---------------- GENERAL CHAT ----------------
if prompt := st.chat_input("Type a message or ask for a PPT..."):
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
    st.subheader(f"📝 Preview Outline: {outline.get('title','Untitled')}")

    for idx, slide in enumerate(outline.get("slides", []), start=1):
        with st.expander(f"Slide {idx}: {slide['title']}", expanded=False):
            st.markdown(slide["description"].replace("\n", "\n\n"))

    new_title = st.text_input("📌 Edit Title", value=outline.get("title", "Untitled"))
    feedback_box = st.text_area("✏️ Feedback for outline (optional):")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback"):
            with st.spinner("Updating outline..."):
                edit_payload = {"outline": outline, "feedback": feedback_box}
                resp = requests.post(f"{BACKEND_URL}/edit-ppt-outline", json=edit_payload, timeout=120)
                if resp.status_code == 200:
                    updated_outline = resp.json()
                    updated_outline["title"] = new_title.strip()
                    st.session_state.outline_chat = updated_outline
                    st.success("✅ Outline updated!")
                    st.rerun()
                else:
                    st.error(f"❌ Edit failed: {resp.text}")

    with col2:
        if st.button("✅ Generate PPT"):
            with st.spinner("Generating PPT..."):
                outline_to_send = copy.deepcopy(outline)
                outline_to_send["title"] = new_title.strip()
                resp = requests.post(f"{BACKEND_URL}/generate-ppt", json={"outline": outline_to_send}, timeout=180)
                if resp.status_code == 200:
                    filename = resp.headers.get("content-disposition","").split("filename=")[-1].strip('"') or "presentation.pptx"
                    st.success("✅ PPT generated successfully!")
                    st.download_button(
                        "⬇️ Download PPT",
                        data=resp.content,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    st.session_state.generated_files.append({
                        "type": "ppt",
                        "filename": filename,
                        "content": resp.content,
                    })
                    st.session_state.outline_chat = None
                else:
                    st.error(f"❌ Generation failed: {resp.text}")


# ---------------- DOC UPLOAD SECTION ----------------
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


# ---------------- CHAT WITH DOCUMENT ----------------
if st.session_state.summary_text:
    st.markdown("💬 **Chat with your uploaded document**")

    for role, content in st.session_state.doc_chat_history:
        with st.chat_message(role):
            st.markdown(content)

    if doc_prompt := st.chat_input("Ask a question about the uploaded document..."):
        st.session_state.doc_chat_history.append(("user", doc_prompt))
        text = doc_prompt.lower()

        try:
            if "ppt" in text or "presentation" in text or "slides" in text:
                with st.spinner("Generating PPT outline from document..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/generate-ppt-outline",
                        json={"description": st.session_state.summary_text + "\n\n" + doc_prompt},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        outline_data = resp.json()
                        outline_data["title"] = st.session_state.summary_title
                        st.session_state.outline_from_summary = outline_data
                        st.session_state.doc_chat_history.append(("assistant", "✅ Generated PPT outline from document. Preview below."))
                    else:
                        st.session_state.doc_chat_history.append(("assistant", f"❌ PPT outline failed: {resp.text}"))
            else:
                resp = requests.post(
                    f"{BACKEND_URL}/chat-doc",
                    json={"message": doc_prompt, "document_text": st.session_state.summary_text},
                    timeout=120,
                )
                if resp.status_code == 200:
                    answer = resp.json().get("response", "⚠️ No answer")
                else:
                    answer = f"❌ Error: {resp.status_code} — {resp.text}"
                st.session_state.doc_chat_history.append(("assistant", answer))

        except Exception as e:
            st.session_state.doc_chat_history.append(("assistant", f"⚠️ Backend error: {e}"))

        st.rerun()


# ---------------- OUTLINE FROM UPLOAD ----------------
if st.session_state.outline_from_summary:
    outline_preview = st.session_state.outline_from_summary
    st.subheader(f"📝 Preview Outline from Document: {outline_preview.get('title','Untitled')}")

    for idx, slide in enumerate(outline_preview.get("slides", []), start=1):
        with st.expander(f"Slide {idx}: {slide['title']}", expanded=False):
            st.markdown(slide["description"].replace("\n", "\n\n"))

    new_title_upload = st.text_input("📌 Edit Title (Upload Flow)", value=outline_preview.get("title", "Untitled"))
    feedback_box = st.text_area("✏️ Feedback for outline (optional):")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback (Upload Flow)"):
            with st.spinner("Applying feedback..."):
                edit_payload = {"outline": outline_preview, "feedback": feedback_box}
                edit_resp = requests.post(f"{BACKEND_URL}/edit-ppt-outline", json=edit_payload, timeout=120)
                if edit_resp.status_code == 200:
                    updated_outline = edit_resp.json()
                    updated_outline["title"] = new_title_upload.strip()
                    st.session_state.outline_from_summary = updated_outline
                    st.success("✅ Outline updated")
                else:
                    st.error(f"❌ Edit failed: {edit_resp.status_code} — {edit_resp.text}")

    with col2:
        if st.button("✅ Generate PPT (Upload Flow)"):
            with st.spinner("Generating PPT..."):
                outline_to_send = copy.deepcopy(outline_preview)
                outline_to_send["title"] = new_title_upload.strip()
                file_resp = requests.post(f"{BACKEND_URL}/generate-ppt", json={"outline": outline_to_send}, timeout=180)
                if file_resp.status_code == 200:
                    filename = file_resp.headers.get("content-disposition","").split("filename=")[-1].strip('"') or "presentation.pptx"
                    st.success("✅ PPT generated successfully!")
                    st.download_button(
                        "⬇️ Download PPT",
                        data=file_resp.content,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key=f"download_upload_ppt_{filename}"
                    )
                    st.session_state.generated_files.append({
                        "type": "ppt",
                        "filename": filename,
                        "content": file_resp.content,
                    })
                else:
                    st.error(f"❌ Generation failed: {file_resp.status_code} — {file_resp.text}")


ppt_generator.py
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def create_ppt(title: str, slides: list, filename: str, branding: dict = None):
    """
    Generate a PowerPoint file styled similar to Dr. Reddy's corporate presentations.
    """

    prs = Presentation()

    # Default branding
    if branding is None:
        branding = {
            "primary_color": (83, 27, 147),      # Dr. Reddy’s purple
            "accent_color": (255, 102, 0),       # Orange accent
            "font_name": "Calibri",
            "logo_path": None,
            "title_font_size": Pt(44),
            "heading_font_size": Pt(28),
            "body_font_size": Pt(18),
        }

    # ---- Title Slide ----
    title_slide_layout = prs.slide_layouts[0]
    slide0 = prs.slides.add_slide(title_slide_layout)
    title_shape = slide0.shapes.title
    subtitle_shape = slide0.placeholders[1]

    title_shape.text = title
    tpara = title_shape.text_frame.paragraphs[0]
    tpara.font.size = branding["title_font_size"]
    tpara.font.bold = True
    tpara.font.color.rgb = RGBColor(*branding["primary_color"])
    tpara.font.name = branding["font_name"]

    subtitle_shape.text = "Corporate Presentation"
    spara = subtitle_shape.text_frame.paragraphs[0]
    spara.font.size = Pt(20)
    spara.font.color.rgb = RGBColor(100, 100, 100)

    if branding.get("logo_path"):
        try:
            left = prs.slide_width - Inches(2)
            top = Inches(0.3)
            height = Inches(1)
            slide0.shapes.add_picture(branding["logo_path"], left, top, height=height)
        except Exception as e:
            print(f"⚠️ Could not add logo: {e}")

    # ---- Content Slides ----
    content_layout = prs.slide_layouts[1]

    for s in slides:
        slide = prs.slides.add_slide(content_layout)

        # Title
        title_sh = slide.shapes.title
        title_sh.text = s["title"]
        para = title_sh.text_frame.paragraphs[0]
        para.font.size = branding["heading_font_size"]
        para.font.bold = True
        para.font.color.rgb = RGBColor(*branding["primary_color"])
        para.font.name = branding["font_name"]

        # Body
        content_sh = slide.shapes.placeholders[1]
        content_tf = content_sh.text_frame
        content_tf.clear()

        for line in s["description"].split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            p = content_tf.add_paragraph()
            p.text = stripped
            p.font.size = branding["body_font_size"]
            p.font.color.rgb = RGBColor(60, 60, 60)
            p.font.name = branding["font_name"]
            p.level = 0

    # ---- Footer with slide numbers ----
    for i, slide in enumerate(prs.slides):
        left = prs.slide_width - Inches(1)
        top = prs.slide_height - Inches(0.5)
        txBox = slide.shapes.add_textbox(left, top, Inches(1), Inches(0.3))
        tf = txBox.text_frame
        p = tf.paragraphs[0]
        p.text = str(i)
        p.font.size = Pt(12)
        p.font.color.rgb = RGBColor(150, 150, 150)
        p.font.name = branding["font_name"]

    prs.save(filename)
    return filename









