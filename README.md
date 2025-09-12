from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, re, datetime, tempfile, fitz, docx
from ppt_generator import create_ppt
from google import genai

# ---------------- CONFIG ----------------
API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")  # set in env or replace here
MODEL_NAME = "gemini-2.0-flash"  # or gemini-1.5-pro, etc.

client = genai.Client(api_key=API_KEY)

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
def extract_slide_count(description: str, default: int = 5) -> int:
    m = re.search(r"(\d+)\s*slides?", description, re.IGNORECASE)
    if m:
        total = int(m.group(1))
        return max(1, total - 1)
    return default - 1

def generate_title(summary: str) -> str:
    prompt = f"""Read the following summary and create a short, clear, presentation-style title.
- Keep it under 12 words
- Do not include birth dates, long sentences, or excessive details
- Just give a clean title, like a presentation heading

Summary:
{summary}
"""
    return call_gemini(prompt).strip()


@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path, file.filename)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Unsupported, empty, or unreadable file content.")

    try:
        summary = summarize_long_text(text)

        # 🔥 Instead of infer_title → use Gemini to generate a nice title
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


def infer_title(description: str) -> str:
    description = description.strip()
    m = re.search(
        r"(?:ppt|presentation)\s+on\s+([A-Za-z0-9\s.&'\-]{2,80})",
        description,
        re.IGNORECASE,
    )
    if m:
        return re.sub(r"\s+in\s+\d+\s+slides?$", "", m.group(1).strip(), flags=re.IGNORECASE)
    return description.title() or "Presentation"

def parse_points(points_text: str):
    points = []
    current_title, current_bullets = None, []
    text = points_text.replace("•", "- ").replace("–", "- ")
    lines = [re.sub(r"[#*>`]", "", ln).strip() for ln in text.splitlines()]

    for line in lines:
        if not line or "Would you like" in line:
            continue
        m = re.match(r"^\s*Slide\s*(\d+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            if current_title:
                points.append({"title": current_title, "description": "\n".join(current_bullets)})
            current_title, current_bullets = m.group(2).strip(), []
            continue
        if line.startswith("-"):
            bullet_text = line.lstrip("-").strip()
            if bullet_text:
                current_bullets.append(bullet_text)
        else:
            if line.strip():
                current_bullets.append(line.strip())
    if current_title:
        points.append({"title": current_title, "description": "\n".join(current_bullets)})
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

# ---------------- Gemini Calls ----------------
def call_gemini(prompt: str) -> str:
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return resp.text.strip()

def generate_outline_from_desc(description: str, num_slides: int):
    prompt = f"""Create a PowerPoint outline on: {description}.
    Generate exactly {num_slides} content slides (excluding title slide).
    Format strictly like this:
    Slide 1: <Title>
    - Bullet
    - Bullet
    - Bullet
    """
    points_text = call_gemini(prompt)
    return parse_points(points_text)

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

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

def clean_title(title: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", title).strip()

# ---------------- ROUTES ----------------
@app.post("/chat")
def chat(req: ChatRequest):
    if "ppt" in req.message.lower() or "presentation" in req.message.lower():
        return {"response": "📑 I can help you create a PPT! Tell me more details (topic, slides, etc.)."}
    reply = call_gemini(req.message)
    return {"response": reply}

@app.post("/generate-outline")
def generate_outline(request: GeneratePPTRequest):
    title = infer_title(request.description)
    num_content_slides = extract_slide_count(request.description, default=5)
    points = generate_outline_from_desc(request.description, num_content_slides)
    return {"title": title, "slides": points}

@app.post("/edit-outline")
def edit_outline(request: EditRequest):
    outline_text = ""
    for idx, slide in enumerate(request.outline.slides, start=1):
        outline_text += f"Slide {idx}: {slide.title}\n"
        for bullet in slide.description.split("\n"):
            outline_text += f"- {bullet}\n"
    prompt = f"""You are editing a PowerPoint outline.
    Here is the outline:
    {outline_text}
    Feedback: "{request.feedback}"
    Update the outline according to the feedback and return in the same format.
    """
    points_text = call_gemini(prompt)
    points = parse_points(points_text)
    return {"title": request.outline.title, "slides": points}

@app.post("/generate-ppt")
def generate_ppt(req: GeneratePPTRequest):
    if req.outline:
        title = clean_title(req.outline.title)
        if len(title) > 80:
            title = "Presentation"
        points = [{"title": clean_title(s.title), "description": s.description} for s in req.outline.slides]
    else:
        title = clean_title(infer_title(req.description))
        if len(title) > 80:
            title = "Presentation"
        num_content_slides = extract_slide_count(req.description, default=5)
        points = generate_outline_from_desc(req.description, num_content_slides)

    
    safe_title = sanitize_filename(title.replace(" ", "_"))
    if len(safe_title) > 100:
        safe_title = "presentation"
    filename = f"{safe_title}.pptx"
    create_ppt(title, points, filename=filename)

    return FileResponse(filename, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation", filename=filename)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = extract_text(tmp_path, file.filename)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Unsupported, empty, or unreadable file content.")

    try:
        summary = summarize_long_text(text)
        title = infer_title(summary) or os.path.splitext(file.filename)[0]
        return {
            "filename": file.filename,
            "chars": len(text),
            "chunks": len(split_text(text)),
            "title": title,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")
