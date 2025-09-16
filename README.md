←[32mINFO←[0m:     Waiting for application shutdown.
←[32mINFO←[0m:     Application shutdown complete.
←[32mINFO←[0m:     Finished server process [←[36m19060←[0m]
C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\vertexai\generative_models\_generative_models.py:433: UserWarning: This feature is deprecated as of June 24, 2025 and will be removed on June 24, 2026. For details, see https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk.
  warning_logs.show_deprecation_warning()
C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\vertexai\_model_garden\_model_garden_models.py:278: UserWarning: This feature is deprecated as of June 24, 2025 and will be removed on June 24, 2026. For details, see https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk.
  warning_logs.show_deprecation_warning()
Process SpawnProcess-18:
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\multiprocessing\process.py", line 315, in _bootstrap
    self.run()
  File "C:\Program Files\Python310\lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\uvicorn\_subprocess.py", line 80, in subprocess_started
    target(sockets=sockets)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\uvicorn\server.py", line 67, in run
    return asyncio.run(self.serve(sockets=sockets))
  File "C:\Program Files\Python310\lib\asyncio\runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "C:\Program Files\Python310\lib\asyncio\base_events.py", line 641, in run_until_complete
    return future.result()
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\uvicorn\server.py", line 71, in serve
    await self._serve(sockets)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\uvicorn\server.py", line 78, in _serve
    config.load()
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\uvicorn\config.py", line 436, in load
    self.loaded_app = import_from_string(self.app)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\uvicorn\importer.py", line 19, in import_from_string
    module = importlib.import_module(module_str)
  File "C:\Program Files\Python310\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "C:\Users\p90022569\Downloads\vertex\main.py", line 25, in <module>
    IMAGE_MODEL = ImageGenerationModel.from_pretrained(IMAGE_MODEL_NAME)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\vertexai\_model_garden\_model_garden_models.py", line 290, in from_pretrained
    return _from_pretrained(interface_class=cls, model_name=model_name)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\vertexai\_model_garden\_model_garden_models.py", line 207, in _from_pretrained
    model_info = _get_model_info(
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\vertexai\_model_garden\_model_garden_models.py", line 123, in _get_model_info
    _publisher_models._PublisherModel(  # pylint: disable=protected-access
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\cloud\aiplatform\_publisher_models.py", line 77, in __init__
    self._gca_resource = getattr(self.api_client, self._getter_method)(
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\cloud\aiplatform_v1\services\model_garden_service\client.py", line 927, in get_publisher_model
    response = rpc(
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\api_core\gapic_v1\method.py", line 131, in __call__
    return wrapped_func(*args, **kwargs)
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\api_core\retry\retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\api_core\retry\retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\api_core\retry\retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\api_core\retry\retry_unary.py", line 147, in retry_target
    result = target()
  File "C:\Users\p90022569\Downloads\vertex\venv\lib\site-packages\google\api_core\grpc_helpers.py", line 78, in error_remapped_callable
    raise exceptions.from_grpc_error(exc) from exc
google.api_core.exceptions.NotFound: 404 Publisher Model `publishers/google/models/imagen-4.0-ultra` is not found.
