main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, re, requests

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

app.py
import copy
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI PPT Generator", layout="wide")
st.title("💡 Chatbot (PPT Generator)")

if "messages" not in st.session_state: st.session_state.messages = []
if "outline_chat" not in st.session_state: st.session_state.outline_chat = None
if "generated_files" not in st.session_state: st.session_state.generated_files = []


# ---- Chat history ----
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# ---- Past generated PPTs ----
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

# ---- Chat input ----
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

# ---- Outline Preview ----
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

def health():
    return {"status": "ok", "text_model": "gemini-2.0-flash"}
