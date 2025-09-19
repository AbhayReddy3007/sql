This is my main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, re, datetime, tempfile, fitz, docx, base64

from ppt_generator import create_ppt
from doc_generator import create_doc

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel

# ---------------- CONFIG ----------------
PROJECT_ID = "drl-zenai-prod"  
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

TEXT_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL = GenerativeModel(TEXT_MODEL_NAME)

IMAGE_MODEL_NAME = "imagen-4.0-generate-001"  
IMAGE_MODEL = ImageGenerationModel.from_pretrained(IMAGE_MODEL_NAME)

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

class DocOutline(BaseModel):
    title: str
    sections: List[Section]

class EditRequest(BaseModel):
    outline: Outline
    feedback: str

class EditDocRequest(BaseModel):
    outline: DocOutline
    feedback: str

class GeneratePPTRequest(BaseModel):
    description: str = ""
    outline: Optional[Outline] = None

class GenerateDocRequest(BaseModel):
    description: str = ""
    outline: Optional[DocOutline] = None

class ImageRequest(BaseModel):
    prompt: str


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


def generate_images_for_points(points, mode="ppt"):
    """Generate images only for slides/sections where it's useful.
    It is not needed for each and every slide.
    Not all slides should have images."""
    images = []
    for idx, item in enumerate(points, start=1):
        if should_generate_image(item['title'], item['description']):
            img_prompt = (
                f"An illustration for a {mode.upper()} section titled '{item['title']}'. "
                f"Content: {item['description']}. "
                f"Style: professional, modern, clean, infographic look."
            )
            try:
                resp = IMAGE_MODEL.generate_images(prompt=img_prompt, number_of_images=1)
                if resp.images and hasattr(resp.images[0], "_image_bytes"):
                    img_bytes = resp.images[0]._image_bytes
                else:
                    img_bytes = None

                if img_bytes:
                    img_path = save_temp_image(img_bytes, idx, item["title"])
                    images.append(img_path)
                else:
                    images.append(None)
            except Exception as e:
                print(f"⚠️ Image generation failed for {mode} {idx}: {e}")
                images.append(None)
        else:
            images.append(None)
    return images


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

    images = generate_images_for_points(points, mode="ppt")

    output_dir = os.path.join(os.path.dirname(__file__), "generated_files")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{sanitize_filename(title)}.pptx")

    create_ppt(title, points, filename=filename, images=images)

    return FileResponse(filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=os.path.basename(filename)
    )

@app.post("/generate-doc-outline")
def generate_doc_outline(request: GenerateDocRequest):
    title = generate_title(request.description)
    num_sections = extract_slide_count(request.description, default=None)
    points = generate_outline_from_desc(request.description, num_sections, mode="doc")
    return {"title": title, "sections": points}

@app.post("/generate-doc")
def generate_doc(req: GenerateDocRequest):
    if req.outline:
        title = clean_title(req.outline.title) or "Document"
        points = [{"title": clean_title(s.title), "description": s.description} for s in req.outline.sections]
    else:
        title = clean_title(generate_title(req.description))
        num_sections = extract_slide_count(req.description, default=None)
        points = generate_outline_from_desc(req.description, num_sections, mode="doc")

    images = generate_images_for_points(points, mode="doc")

    output_dir = os.path.join(os.path.dirname(__file__), "generated_files")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{sanitize_filename(title)}.docx")

    create_doc(title, points, filename=filename, images=images)

    return FileResponse(filename,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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

@app.post("/generate-image")
def generate_image(req: ImageRequest):
    try:
        resp = IMAGE_MODEL.generate_images(prompt=req.prompt, number_of_images=1)

        if resp.images and hasattr(resp.images[0], "_image_bytes"):
            img_bytes = resp.images[0]._image_bytes
        else:
            raise HTTPException(status_code=500, detail="Image generation failed")

        output_dir = os.path.join(os.path.dirname(__file__), "generated_files", "images")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"generated_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        with open(filename, "wb") as f:
            f.write(img_bytes)

        return FileResponse(filename, media_type="image/png", filename=os.path.basename(filename))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "text_model": TEXT_MODEL_NAME, "image_model": IMAGE_MODEL_NAME}

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

@app.post("/edit-doc-outline")
def edit_doc_outline(req: EditDocRequest):
    outline_text = "\n".join(
        [f"Section {i+1}: {s.title}\n{s.description}" for i, s in enumerate(req.outline.sections)]
    )
    prompt = f"""
    You are an assistant improving a Document outline.

    Current Outline:
    Title: {req.outline.title}
    {outline_text}

    Feedback:
    {req.feedback}

    Task:
    - Apply the feedback to refine/improve the outline.
    - Return the updated outline with the same format:
      Section 1: <Title>
      <Paragraph 1>
      <Paragraph 2>
      <Paragraph 3>
    - Avoid bullet points, use prose.
    """
    try:
        updated_points = parse_points(call_vertex(prompt))
        return {"title": req.outline.title, "sections": updated_points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Doc outline editing failed: {e}")

This is my app.py
import copy
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"  

st.set_page_config(page_title="AI Productivity Suite", layout="wide")
st.title("Chatbot")

# ---------------- Helpers ----------------
def extract_filename_from_cd(resp):
    cd = resp.headers.get("content-disposition", "")
    if "filename=" in cd:
        return cd.split("filename=")[-1].strip().strip('"')
    return None

def render_outline_preview(outline_data, mode="ppt"):
    if not outline_data:
        st.info("No outline available.")
        return False

    title = outline_data.get("title", "Untitled")
    items = outline_data.get("slides", []) if mode == "ppt" else outline_data.get("sections", [])
    st.subheader(f"📝 Preview Outline: {title}")

    for idx, item in enumerate(items, start=1):
        item_title = item.get("title", f"{'Slide' if mode=='ppt' else 'Section'} {idx}")
        item_desc = item.get("description", "")
        with st.expander(f"{'Slide' if mode=='ppt' else 'Section'} {idx}: {item_title}", expanded=False):
            st.markdown(item_desc.replace("\n", "\n\n"))
    return len(items) > 0


# ---------------- STATE ----------------
defaults = {
    "messages": [],
    "outline_chat": None,
    "outline_mode": None,  # "ppt" or "doc"
    "generated_files": [],
    "summary_text": None,
    "summary_title": None,
    "doc_chat_history": [],
    "outline_from_summary": None,
    "generated_images": [],  # store past generated images
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------- CHAT HISTORY ----------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# ---------------- Past Generated Files ----------------
for i, file_info in enumerate(st.session_state.generated_files):
    with st.chat_message("assistant"):
        if file_info["type"] == "ppt":
            st.markdown("✅ PPT generated earlier! Download below:")
            st.download_button(
                "⬇️ Download PPT",
                data=file_info["content"] if file_info["content"] else b"",
                file_name=file_info["filename"],
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key=f"past_download_ppt_{i}"
            )
        elif file_info["type"] == "doc":
            st.markdown("✅ Document generated earlier! Download below:")
            st.download_button(
                "⬇️ Download Document",
                data=file_info["content"] if file_info["content"] else b"",
                file_name=file_info["filename"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"past_download_doc_{i}"
            )

# ---------------- Past Generated Images ----------------
for i, img_info in enumerate(st.session_state.generated_images):
    with st.chat_message("assistant"):
        st.markdown("🖼️ Image generated earlier:")
        st.image(img_info["content"], caption=img_info["filename"], use_container_width=True)
        st.download_button(
            "⬇️ Download Image",
            data=img_info["content"],
            file_name=img_info["filename"],
            mime="image/png",
            key=f"past_download_img_{i}"
        )

# ---------------- GENERAL CHAT ----------------
if prompt := st.chat_input("Type a message, ask for a PPT or DOC..."):
    st.session_state.messages.append(("user", prompt))
    text = prompt.lower()

    try:
        if "ppt" in text or "presentation" in text or "slides" in text:
            with st.spinner("Generating PPT outline..."):
                resp = requests.post(f"{BACKEND_URL}/generate-ppt-outline", json={"description": prompt}, timeout=120)
                if resp.status_code == 200:
                    st.session_state.outline_chat = resp.json()
                    st.session_state.outline_mode = "ppt"
                    st.session_state.messages.append(("assistant", "✅ PPT outline generated! Preview below."))
                else:
                    st.session_state.messages.append(("assistant", f"❌ PPT outline failed: {resp.text}"))

        elif "doc" in text or "document" in text or "report" in text or "pages" in text or "sections" in text:
            with st.spinner("Generating DOC outline..."):
                resp = requests.post(f"{BACKEND_URL}/generate-doc-outline", json={"description": prompt}, timeout=120)
                if resp.status_code == 200:
                    st.session_state.outline_chat = resp.json()
                    st.session_state.outline_mode = "doc"
                    st.session_state.messages.append(("assistant", "✅ DOC outline generated! Preview below."))
                else:
                    st.session_state.messages.append(("assistant", f"❌ DOC outline failed: {resp.text}"))

        elif "image" in text or "picture" in text or "photo" in text:
            with st.spinner("Generating Image..."):
                resp = requests.post(f"{BACKEND_URL}/generate-image", json={"prompt": prompt}, timeout=180)
                if resp.status_code == 200:
                    img_bytes = resp.content
                    filename = extract_filename_from_cd(resp) or f"image_{len(st.session_state.generated_images)+1}.png"

                    # Store image
                    st.session_state.generated_images.append({
                        "filename": filename,
                        "content": img_bytes,
                    })

                    # Show image directly
                    st.image(img_bytes, caption=filename, use_container_width=True)

                    # Download button
                    st.download_button(
                        "⬇️ Download Image",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/png",
                        key=f"download_img_{filename}"
                    )

                    st.session_state.messages.append(("assistant", "✅ Image generated!"))
                else:
                    st.session_state.messages.append(("assistant", f"❌ Image generation failed: {resp.text}"))

        else:
            resp = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt}, timeout=60)
            bot_reply = resp.json().get("response", "⚠️ Error")
            st.session_state.messages.append(("assistant", bot_reply))

    except Exception as e:
        st.session_state.messages.append(("assistant", f"⚠️ Backend error: {e}"))

    st.rerun()

# ---------------- OUTLINE PREVIEW + ACTIONS ----------------
if st.session_state.outline_chat:
    mode = st.session_state.outline_mode
    outline = st.session_state.outline_chat

    render_outline_preview(outline, mode=mode)

    new_title = st.text_input("📌 Edit Title", value=outline.get("title", "Untitled"), key=f"title_{mode}")
    feedback_box = st.text_area("✏️ Feedback for outline (optional):", value="", key=f"feedback_chat_{mode}")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback"):
            with st.spinner("Updating outline..."):
                try:
                    edit_payload = {"outline": outline, "feedback": feedback_box}
                    endpoint = f"{BACKEND_URL}/edit-ppt-outline" if mode == "ppt" else f"{BACKEND_URL}/edit-doc-outline"
                    resp = requests.post(endpoint, json=edit_payload, timeout=120)
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
        if st.button(f"✅ Generate {mode.upper()}"):
            with st.spinner(f"Generating {mode.upper()}..."):
                try:
                    outline_to_send = copy.deepcopy(outline)
                    outline_to_send["title"] = new_title.strip() if new_title else outline_to_send["title"]

                    endpoint = f"{BACKEND_URL}/generate-ppt" if mode == "ppt" else f"{BACKEND_URL}/generate-doc"
                    resp = requests.post(endpoint, json={"outline": outline_to_send}, timeout=180)
                    if resp.status_code == 200:
                        filename = extract_filename_from_cd(resp) or (
                            "presentation.pptx" if mode == "ppt" else "document.docx"
                        )

                        st.success(f"✅ {mode.upper()} generated successfully!")
                        st.download_button(
                            f"⬇️ Download {mode.upper()}",
                            data=resp.content if resp.content else b"",
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                                 if mode == "ppt"
                                 else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"download_{mode}_{filename}"
                        )

                        st.session_state.generated_files.append({
                            "type": mode,
                            "filename": filename,
                            "content": resp.content,
                        })

                        st.session_state.outline_chat = None
                    else:
                        st.error(f"❌ Generation failed: {resp.status_code} — {resp.text}")
                except Exception as e:
                    st.error(f"❌ Generation error: {e}")


# ---------------- DOC UPLOAD SECTION ----------------
uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt", "md"])

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
                        st.session_state.outline_mode = "ppt"
                        st.session_state.doc_chat_history.append(("assistant", "✅ Generated PPT outline from document. Preview below."))
                    else:
                        st.session_state.doc_chat_history.append(("assistant", f"❌ PPT outline failed: {resp.text}"))

            elif "doc" in text or "document" in text or "report" in text or "pages" in text or "sections" in text:
                with st.spinner("Generating DOC outline from document..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/generate-doc-outline",
                        json={"description": st.session_state.summary_text + "\n\n" + doc_prompt},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        outline_data = resp.json()
                        outline_data["title"] = st.session_state.summary_title
                        st.session_state.outline_from_summary = outline_data
                        st.session_state.outline_mode = "doc"
                        st.session_state.doc_chat_history.append(("assistant", "✅ Generated DOC outline from document. Preview below."))
                    else:
                        st.session_state.doc_chat_history.append(("assistant", f"❌ DOC outline failed: {resp.text}"))

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
    mode = st.session_state.outline_mode
    outline_preview = st.session_state.outline_from_summary

    render_outline_preview(outline_preview, mode=mode)

    new_title_upload = st.text_input("📌 Edit Title (Upload Flow)", value=outline_preview.get("title", "Untitled"), key=f"title_upload_{mode}")
    feedback_box = st.text_area("✏️ Feedback for outline (optional):", value="", key=f"feedback_upload_{mode}")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback (Upload Flow)"):
            with st.spinner("Applying feedback..."):
                edit_payload = {"outline": outline_preview, "feedback": feedback_box}
                endpoint = f"{BACKEND_URL}/edit-ppt-outline" if mode == "ppt" else f"{BACKEND_URL}/edit-doc-outline"
                edit_resp = requests.post(endpoint, json=edit_payload, timeout=120)
                if edit_resp.status_code == 200:
                    updated_outline = edit_resp.json()
                    updated_outline["title"] = new_title_upload.strip() if new_title_upload else updated_outline["title"]
                    st.session_state.outline_from_summary = updated_outline
                    st.success("✅ Outline updated")
                else:
                    st.error(f"❌ Edit failed: {edit_resp.status_code} — {edit_resp.text}")

    with col2:
        if st.button(f"✅ Generate {mode.upper()} (Upload Flow)"):
            with st.spinner(f"Generating {mode.upper()}..."):
                outline_to_send = copy.deepcopy(outline_preview)
                outline_to_send["title"] = new_title_upload.strip() if new_title_upload else outline_to_send["title"]

                endpoint = f"{BACKEND_URL}/generate-ppt" if mode == "ppt" else f"{BACKEND_URL}/generate-doc"
                file_resp = requests.post(endpoint, json={"outline": outline_to_send}, timeout=180)
                if file_resp.status_code == 200:
                    filename = extract_filename_from_cd(file_resp) or (
                        "presentation.pptx" if mode == "ppt" else "document.docx"
                    )

                    st.success(f"✅ {mode.upper()} generated successfully!")
                    st.download_button(
                        f"⬇️ Download {mode.upper()}",
                        data=file_resp.content if file_resp.content else b"",
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                             if mode == "ppt"
                             else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_upload_{mode}_{filename}"
                    )

                    st.session_state.generated_files.append({
                        "type": mode,
                        "filename": filename,
                        "content": file_resp.content,
                    })
                else:
                    st.error(f"❌ Generation failed: {file_resp.status_code} — {file_resp.text}")
This is my ppt_generator.py
import copy
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"  

st.set_page_config(page_title="AI Productivity Suite", layout="wide")
st.title("Chatbot")

# ---------------- Helpers ----------------
def extract_filename_from_cd(resp):
    cd = resp.headers.get("content-disposition", "")
    if "filename=" in cd:
        return cd.split("filename=")[-1].strip().strip('"')
    return None

def render_outline_preview(outline_data, mode="ppt"):
    if not outline_data:
        st.info("No outline available.")
        return False

    title = outline_data.get("title", "Untitled")
    items = outline_data.get("slides", []) if mode == "ppt" else outline_data.get("sections", [])
    st.subheader(f"📝 Preview Outline: {title}")

    for idx, item in enumerate(items, start=1):
        item_title = item.get("title", f"{'Slide' if mode=='ppt' else 'Section'} {idx}")
        item_desc = item.get("description", "")
        with st.expander(f"{'Slide' if mode=='ppt' else 'Section'} {idx}: {item_title}", expanded=False):
            st.markdown(item_desc.replace("\n", "\n\n"))
    return len(items) > 0


# ---------------- STATE ----------------
defaults = {
    "messages": [],
    "outline_chat": None,
    "outline_mode": None,  # "ppt" or "doc"
    "generated_files": [],
    "summary_text": None,
    "summary_title": None,
    "doc_chat_history": [],
    "outline_from_summary": None,
    "generated_images": [],  # store past generated images
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------- CHAT HISTORY ----------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# ---------------- Past Generated Files ----------------
for i, file_info in enumerate(st.session_state.generated_files):
    with st.chat_message("assistant"):
        if file_info["type"] == "ppt":
            st.markdown("✅ PPT generated earlier! Download below:")
            st.download_button(
                "⬇️ Download PPT",
                data=file_info["content"] if file_info["content"] else b"",
                file_name=file_info["filename"],
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key=f"past_download_ppt_{i}"
            )
        elif file_info["type"] == "doc":
            st.markdown("✅ Document generated earlier! Download below:")
            st.download_button(
                "⬇️ Download Document",
                data=file_info["content"] if file_info["content"] else b"",
                file_name=file_info["filename"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"past_download_doc_{i}"
            )

# ---------------- Past Generated Images ----------------
for i, img_info in enumerate(st.session_state.generated_images):
    with st.chat_message("assistant"):
        st.markdown("🖼️ Image generated earlier:")
        st.image(img_info["content"], caption=img_info["filename"], use_container_width=True)
        st.download_button(
            "⬇️ Download Image",
            data=img_info["content"],
            file_name=img_info["filename"],
            mime="image/png",
            key=f"past_download_img_{i}"
        )

# ---------------- GENERAL CHAT ----------------
if prompt := st.chat_input("Type a message, ask for a PPT or DOC..."):
    st.session_state.messages.append(("user", prompt))
    text = prompt.lower()

    try:
        if "ppt" in text or "presentation" in text or "slides" in text:
            with st.spinner("Generating PPT outline..."):
                resp = requests.post(f"{BACKEND_URL}/generate-ppt-outline", json={"description": prompt}, timeout=120)
                if resp.status_code == 200:
                    st.session_state.outline_chat = resp.json()
                    st.session_state.outline_mode = "ppt"
                    st.session_state.messages.append(("assistant", "✅ PPT outline generated! Preview below."))
                else:
                    st.session_state.messages.append(("assistant", f"❌ PPT outline failed: {resp.text}"))

        elif "doc" in text or "document" in text or "report" in text or "pages" in text or "sections" in text:
            with st.spinner("Generating DOC outline..."):
                resp = requests.post(f"{BACKEND_URL}/generate-doc-outline", json={"description": prompt}, timeout=120)
                if resp.status_code == 200:
                    st.session_state.outline_chat = resp.json()
                    st.session_state.outline_mode = "doc"
                    st.session_state.messages.append(("assistant", "✅ DOC outline generated! Preview below."))
                else:
                    st.session_state.messages.append(("assistant", f"❌ DOC outline failed: {resp.text}"))

        elif "image" in text or "picture" in text or "photo" in text:
            with st.spinner("Generating Image..."):
                resp = requests.post(f"{BACKEND_URL}/generate-image", json={"prompt": prompt}, timeout=180)
                if resp.status_code == 200:
                    img_bytes = resp.content
                    filename = extract_filename_from_cd(resp) or f"image_{len(st.session_state.generated_images)+1}.png"

                    # Store image
                    st.session_state.generated_images.append({
                        "filename": filename,
                        "content": img_bytes,
                    })

                    # Show image directly
                    st.image(img_bytes, caption=filename, use_container_width=True)

                    # Download button
                    st.download_button(
                        "⬇️ Download Image",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/png",
                        key=f"download_img_{filename}"
                    )

                    st.session_state.messages.append(("assistant", "✅ Image generated!"))
                else:
                    st.session_state.messages.append(("assistant", f"❌ Image generation failed: {resp.text}"))

        else:
            resp = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt}, timeout=60)
            bot_reply = resp.json().get("response", "⚠️ Error")
            st.session_state.messages.append(("assistant", bot_reply))

    except Exception as e:
        st.session_state.messages.append(("assistant", f"⚠️ Backend error: {e}"))

    st.rerun()

# ---------------- OUTLINE PREVIEW + ACTIONS ----------------
if st.session_state.outline_chat:
    mode = st.session_state.outline_mode
    outline = st.session_state.outline_chat

    render_outline_preview(outline, mode=mode)

    new_title = st.text_input("📌 Edit Title", value=outline.get("title", "Untitled"), key=f"title_{mode}")
    feedback_box = st.text_area("✏️ Feedback for outline (optional):", value="", key=f"feedback_chat_{mode}")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback"):
            with st.spinner("Updating outline..."):
                try:
                    edit_payload = {"outline": outline, "feedback": feedback_box}
                    endpoint = f"{BACKEND_URL}/edit-ppt-outline" if mode == "ppt" else f"{BACKEND_URL}/edit-doc-outline"
                    resp = requests.post(endpoint, json=edit_payload, timeout=120)
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
        if st.button(f"✅ Generate {mode.upper()}"):
            with st.spinner(f"Generating {mode.upper()}..."):
                try:
                    outline_to_send = copy.deepcopy(outline)
                    outline_to_send["title"] = new_title.strip() if new_title else outline_to_send["title"]

                    endpoint = f"{BACKEND_URL}/generate-ppt" if mode == "ppt" else f"{BACKEND_URL}/generate-doc"
                    resp = requests.post(endpoint, json={"outline": outline_to_send}, timeout=180)
                    if resp.status_code == 200:
                        filename = extract_filename_from_cd(resp) or (
                            "presentation.pptx" if mode == "ppt" else "document.docx"
                        )

                        st.success(f"✅ {mode.upper()} generated successfully!")
                        st.download_button(
                            f"⬇️ Download {mode.upper()}",
                            data=resp.content if resp.content else b"",
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                                 if mode == "ppt"
                                 else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"download_{mode}_{filename}"
                        )

                        st.session_state.generated_files.append({
                            "type": mode,
                            "filename": filename,
                            "content": resp.content,
                        })

                        st.session_state.outline_chat = None
                    else:
                        st.error(f"❌ Generation failed: {resp.status_code} — {resp.text}")
                except Exception as e:
                    st.error(f"❌ Generation error: {e}")


# ---------------- DOC UPLOAD SECTION ----------------
uploaded_file = st.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt", "md"])

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
                        st.session_state.outline_mode = "ppt"
                        st.session_state.doc_chat_history.append(("assistant", "✅ Generated PPT outline from document. Preview below."))
                    else:
                        st.session_state.doc_chat_history.append(("assistant", f"❌ PPT outline failed: {resp.text}"))

            elif "doc" in text or "document" in text or "report" in text or "pages" in text or "sections" in text:
                with st.spinner("Generating DOC outline from document..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/generate-doc-outline",
                        json={"description": st.session_state.summary_text + "\n\n" + doc_prompt},
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        outline_data = resp.json()
                        outline_data["title"] = st.session_state.summary_title
                        st.session_state.outline_from_summary = outline_data
                        st.session_state.outline_mode = "doc"
                        st.session_state.doc_chat_history.append(("assistant", "✅ Generated DOC outline from document. Preview below."))
                    else:
                        st.session_state.doc_chat_history.append(("assistant", f"❌ DOC outline failed: {resp.text}"))

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
    mode = st.session_state.outline_mode
    outline_preview = st.session_state.outline_from_summary

    render_outline_preview(outline_preview, mode=mode)

    new_title_upload = st.text_input("📌 Edit Title (Upload Flow)", value=outline_preview.get("title", "Untitled"), key=f"title_upload_{mode}")
    feedback_box = st.text_area("✏️ Feedback for outline (optional):", value="", key=f"feedback_upload_{mode}")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Apply Feedback (Upload Flow)"):
            with st.spinner("Applying feedback..."):
                edit_payload = {"outline": outline_preview, "feedback": feedback_box}
                endpoint = f"{BACKEND_URL}/edit-ppt-outline" if mode == "ppt" else f"{BACKEND_URL}/edit-doc-outline"
                edit_resp = requests.post(endpoint, json=edit_payload, timeout=120)
                if edit_resp.status_code == 200:
                    updated_outline = edit_resp.json()
                    updated_outline["title"] = new_title_upload.strip() if new_title_upload else updated_outline["title"]
                    st.session_state.outline_from_summary = updated_outline
                    st.success("✅ Outline updated")
                else:
                    st.error(f"❌ Edit failed: {edit_resp.status_code} — {edit_resp.text}")

    with col2:
        if st.button(f"✅ Generate {mode.upper()} (Upload Flow)"):
            with st.spinner(f"Generating {mode.upper()}..."):
                outline_to_send = copy.deepcopy(outline_preview)
                outline_to_send["title"] = new_title_upload.strip() if new_title_upload else outline_to_send["title"]

                endpoint = f"{BACKEND_URL}/generate-ppt" if mode == "ppt" else f"{BACKEND_URL}/generate-doc"
                file_resp = requests.post(endpoint, json={"outline": outline_to_send}, timeout=180)
                if file_resp.status_code == 200:
                    filename = extract_filename_from_cd(file_resp) or (
                        "presentation.pptx" if mode == "ppt" else "document.docx"
                    )

                    st.success(f"✅ {mode.upper()} generated successfully!")
                    st.download_button(
                        f"⬇️ Download {mode.upper()}",
                        data=file_resp.content if file_resp.content else b"",
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                             if mode == "ppt"
                             else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_upload_{mode}_{filename}"
                    )

                    st.session_state.generated_files.append({
                        "type": mode,
                        "filename": filename,
                        "content": file_resp.content,
                    })
                else:
                    st.error(f"❌ Generation failed: {file_resp.status_code} — {file_resp.text}")
