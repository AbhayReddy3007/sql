def call_vertex_ai(prompt: str) -> str:
    credentials = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO
    )
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

    model = GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()
