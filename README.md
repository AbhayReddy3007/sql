(venv) C:\Users\p90022569\Downloads\d2p>pip install --upgrade google-genai
Requirement already satisfied: google-genai in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (1.35.0)
Collecting google-genai
  Downloading google_genai-1.36.0-py3-none-any.whl (244 kB)
     |████████████████████████████████| 244 kB 3.3 MB/s
Requirement already satisfied: websockets<15.1.0,>=13.0.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (15.0.1)
Requirement already satisfied: pydantic<3.0.0,>=2.0.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (2.11.7)
Requirement already satisfied: tenacity<9.2.0,>=8.2.3 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (9.1.2)
Requirement already satisfied: typing-extensions<5.0.0,>=4.11.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (4.15.0)
Requirement already satisfied: requests<3.0.0,>=2.28.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (2.32.5)
Requirement already satisfied: google-auth<3.0.0,>=2.14.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (2.40.3)
Requirement already satisfied: httpx<1.0.0,>=0.28.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (0.28.1)
Requirement already satisfied: anyio<5.0.0,>=4.8.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-genai) (4.10.0)
Requirement already satisfied: idna>=2.8 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from anyio<5.0.0,>=4.8.0->google-genai) (3.10)
Requirement already satisfied: exceptiongroup>=1.0.2 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from anyio<5.0.0,>=4.8.0->google-genai) (1.3.0)
Requirement already satisfied: sniffio>=1.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from anyio<5.0.0,>=4.8.0->google-genai) (1.3.1)
Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-auth<3.0.0,>=2.14.1->google-genai) (4.9.1)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-auth<3.0.0,>=2.14.1->google-genai) (5.5.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from google-auth<3.0.0,>=2.14.1->google-genai) (0.4.2)
Requirement already satisfied: certifi in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from httpx<1.0.0,>=0.28.1->google-genai) (2025.8.3)
Requirement already satisfied: httpcore==1.* in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from httpx<1.0.0,>=0.28.1->google-genai) (1.0.9)
Requirement already satisfied: h11>=0.16 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from httpcore==1.*->httpx<1.0.0,>=0.28.1->google-genai) (0.16.0)
Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3.0.0,>=2.14.1->google-genai) (0.6.1)
Requirement already satisfied: annotated-types>=0.6.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from pydantic<3.0.0,>=2.0.0->google-genai) (0.7.0)
Requirement already satisfied: pydantic-core==2.33.2 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from pydantic<3.0.0,>=2.0.0->google-genai) (2.33.2)
Requirement already satisfied: typing-inspection>=0.4.0 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from pydantic<3.0.0,>=2.0.0->google-genai) (0.4.1)
Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from requests<3.0.0,>=2.28.1->google-genai) (3.4.3)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\p90022569\downloads\d2p\venv\lib\site-packages (from requests<3.0.0,>=2.28.1->google-genai) (2.5.0)
Installing collected packages: google-genai
  Attempting uninstall: google-genai
    Found existing installation: google-genai 1.35.0
    Uninstalling google-genai-1.35.0:
      Successfully uninstalled google-genai-1.35.0
Successfully installed google-genai-1.36.0
WARNING: You are using pip version 21.2.3; however, version 25.2 is available.
You should consider upgrading via the 'C:\Users\p90022569\Downloads\d2p\venv\Scripts\python.exe -m pip install --upgrade pip' command.

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
