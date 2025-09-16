# ---------------- EDIT OUTLINE ROUTES ----------------
@app.post("/edit-ppt-outline")
def edit_ppt_outline(req: EditRequest):
    """
    Refine an existing PPT outline based on user feedback.
    """
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
    """
    Refine an existing Document outline based on user feedback.
    """
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
