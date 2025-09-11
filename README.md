←[32mINFO←[0m:     Application startup complete.
C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\vertexai\generative_models\_generative_models.py:433: UserWarning: This feature is deprecated as of June 24, 2025 and will be removed on June 24, 2026. For details, see https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk.
  warning_logs.show_deprecation_warning()
←[32mINFO←[0m:     127.0.0.1:50747 - "←[1mPOST /chat HTTP/1.1←[0m" ←[91m500 Internal Server Error←[0m
←[31mERROR←[0m:    Exception in ASGI application
Traceback (most recent call last):
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\fastapi\applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\middleware\errors.py", line 186, in __call__
    raise exc
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\middleware\errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\middleware\cors.py", line 85, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\middleware\exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\routing.py", line 78, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\routing.py", line 75, in app
    response = await f(request)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\fastapi\routing.py", line 302, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\fastapi\routing.py", line 215, in run_endpoint_function
    return await run_in_threadpool(dependant.call, **values)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\starlette\concurrency.py", line 38, in run_in_threadpool
    return await anyio.to_thread.run_sync(func)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\anyio\to_thread.py", line 56, in run_sync
    return await get_async_backend().run_sync_in_worker_thread(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\anyio\_backends\_asyncio.py", line 2476, in run_sync_in_worker_thread
    return await future
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\anyio\_backends\_asyncio.py", line 967, in run
    result = context.run(func, *args)
  File "C:\Users\p90022569\Downloads\d2p\main.py", line 177, in chat
    reply = call_vertex_ai(req.message)
  File "C:\Users\p90022569\Downloads\d2p\main.py", line 140, in call_vertex_ai
    response = model.generate_content(prompt)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\vertexai\generative_models\_generative_models.py", line 712, in generate_content
    return self._generate_content(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\vertexai\generative_models\_generative_models.py", line 837, in _generate_content
    gapic_response = self._prediction_client.generate_content(request=request)
  File "C:\Program Files\Python310\lib\functools.py", line 970, in __get__
    val = self.func(instance)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\vertexai\generative_models\_generative_models.py", line 3425, in _prediction_client
    return aiplatform_initializer.global_config.create_client(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\cloud\aiplatform\initializer.py", line 603, in create_client
    "credentials": credentials or self.credentials,
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\cloud\aiplatform\initializer.py", line 403, in credentials
    self._set_project_as_env_var_or_google_auth_default()
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\cloud\aiplatform\initializer.py", line 119, in _set_project_as_env_var_or_google_auth_default
    credentials, _ = google.auth.default(scopes=constants.DEFAULT_AUTHED_SCOPES)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\auth\_default.py", line 651, in default
    credentials, project_id = checker()
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\auth\_default.py", line 644, in <lambda>
    lambda: _get_explicit_environ_credentials(quota_project_id=quota_project_id),
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\auth\_default.py", line 293, in _get_explicit_environ_credentials
    credentials, project_id = load_credentials_from_file(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\auth\_default.py", line 125, in load_credentials_from_file
    raise exceptions.DefaultCredentialsError(
google.auth.exceptions.DefaultCredentialsError: File "C:\Users\p90022569\Downloads\d2p\drl-zenai-prod-a61aa1ebee8a (2).json" was not found.
