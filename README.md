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
    "messages": [],
    "outline_chat": None,
    "generated_files": [],
    "summary_text": None,
    "summary_title": None,
    "doc_chat_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------- CHAT HISTORY ----------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)


# ---------------- GENERAL CHAT ----------------
if prompt := st.chat_input("💬 Type a message, ask for a PPT..."):
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

        try:
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
