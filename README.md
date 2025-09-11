(venv) C:\Users\p90022569\Downloads\d2p>uvicorn main:app --reload
←[32mINFO←[0m:     Will watch for changes in these directories: ['C:\\Users\\p90022569\\Downloads\\d2p']
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://127.0.0.1:8000←[0m (Press CTRL+C to quit)
←[32mINFO←[0m:     Started reloader process [←[36m←[1m9688←[0m] using ←[36m←[1mStatReload←[0m
Process SpawnProcess-1:
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\multiprocessing\process.py", line 315, in _bootstrap
    self.run()
  File "C:\Program Files\Python310\lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\_subprocess.py", line 80, in subprocess_started
    target(sockets=sockets)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\server.py", line 67, in run
    return asyncio.run(self.serve(sockets=sockets))
  File "C:\Program Files\Python310\lib\asyncio\runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "C:\Program Files\Python310\lib\asyncio\base_events.py", line 641, in run_until_complete
    return future.result()
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\server.py", line 71, in serve
    await self._serve(sockets)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\server.py", line 78, in _serve
    config.load()
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\config.py", line 436, in load
    self.loaded_app = import_from_string(self.app)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\uvicorn\importer.py", line 19, in import_from_string
    module = importlib.import_module(module_str)
  File "C:\Program Files\Python310\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "C:\Users\p90022569\Downloads\d2p\main.py", line 17, in <module>
    from google.genai.types import Text, CompletionParams
ImportError: cannot import name 'Text' from 'google.genai.types' (C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\types.py)
←[33mWARNING←[0m:  StatReload detected changes in 'main.py'. Reloading...
 ←[32mINFO←[0m:     Started server process [←[36m14832←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
←[32mINFO←[0m:     127.0.0.1:64164 - "←[1mPOST /chat HTTP/1.1←[0m" ←[91m500 Internal Server Error←[0m
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
  File "C:\Users\p90022569\Downloads\d2p\main.py", line 187, in chat
    reply = call_gemini(req.message)
  File "C:\Users\p90022569\Downloads\d2p\main.py", line 145, in call_gemini
    resp = genai_client.models.generate_content(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\models.py", line 6565, in generate_content
    response = self._generate_content(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\models.py", line 5377, in _generate_content
    response = self._api_client.request(
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\_api_client.py", line 1289, in request
    response = self._request(http_request, http_options, stream=False)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\_api_client.py", line 1109, in _request
    return self._retry(self._request_once, http_request, stream)  # type: ignore[no-any-return]
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\tenacity\__init__.py", line 477, in __call__
    do = self.iter(retry_state=retry_state)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\tenacity\__init__.py", line 378, in iter
    result = action(retry_state)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\tenacity\__init__.py", line 420, in exc_check
    raise retry_exc.reraise()
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\tenacity\__init__.py", line 187, in reraise
    raise self.last_attempt.result()
  File "C:\Program Files\Python310\lib\concurrent\futures\_base.py", line 438, in result
    return self.__get_result()
  File "C:\Program Files\Python310\lib\concurrent\futures\_base.py", line 390, in __get_result
    raise self._exception
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\tenacity\__init__.py", line 480, in __call__
    result = fn(*args, **kwargs)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\_api_client.py", line 1086, in _request_once
    errors.APIError.raise_for_response(response)
  File "C:\Users\p90022569\Downloads\d2p\venv\lib\site-packages\google\genai\errors.py", line 105, in raise_for_response
    raise ClientError(status_code, response_json, response)
google.genai.errors.ClientError: 403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'Generative Language API has not been used in project 305423101018 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=305423101018 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.', 'status': 'PERMISSION_DENIED', 'details': [{'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'SERVICE_DISABLED', 'domain': 'googleapis.com', 'metadata': {'containerInfo': '305423101018', 'consumer': 'projects/305423101018', 'activationUrl': 'https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=305423101018', 'service': 'generativelanguage.googleapis.com', 'serviceTitle': 'Generative Language API'}}, {'@type': 'type.googleapis.com/google.rpc.LocalizedMessage', 'locale': 'en-US', 'message': 'Generative Language API has not been used in project 305423101018 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=305423101018 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.'}, {'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Google developers console API activation', 'url': 'https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview?project=305423101018'}]}]}}
