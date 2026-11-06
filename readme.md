# Yet Another Llama.ccp Launcher

## About
A desktop launcher for llama.cpp / ik_llama I've vibecoded for my own use, and which other people may find of use.
It doesn't have switches for all options, and it's tailored for single gpu use, because that's what fits me - I've found Catapult had too many options - but all others launch arguments can be manually added if needed.
## Features

It creates a launch command for either llama.cpp or ik_llama and scans model directories for GGUF files, parses their headers, estimates VRAM usage (it tends to overestimate context size requirements), and launches a backend server. A FastAPI backend (launcher.py) serves the UI (index.html) over
localhost; app.py opens it in a native web-view window via pywebview.
The GUI allows you to sort models by file or name, filter out those with multimodal support, and indicates if the model is MoE, has mtp, vision or sliding window attention. It has a search box and indicates the model quant and file size.
If there are multiple mmproj files in the same subfolder (ie, comfyui text_encoders folder if you use ggufs), it uses a dropdown menu to pick the proper mmproj


## Requirements
------------

Python
  - Python 3 (3.10+ recommended). A virtual environment is created in `venv`.

Linux system packages (runtime)
  - python3-gi, python3-gi-cairo, gir1.2-webkit2-4.1
    (GTK + WebKit for the pywebview window)

Linux compile dependencies
  These are only needed because PyGObject/pycairo are installed through pip in
  the venv (rather than from distro packages) and must be compiled against the
  following headers:

  | Package                 | Purpose                                              |
  |-------------------------|------------------------------------------------------|
  | libgirepository1.0-dev  | Girepository headers (old API)                       |
  | libgirepository-2.0-dev | Girepository 2.0 headers (new API, required by       |
  |                         | PyGObject >= 3.52)                                   |
  | libcairo2-dev           | Cairo headers, required by pycairo                   |
  | pkg-config              | Library detection during compilation                 |
  | python3-dev             | Python headers (Python.h)                            |

  On Windows these system packages are not available; PyGObject/GTK is not
  packaged for Windows, so the GUI automatically falls back to Qt WebEngine
  (see requirements-windows.txt). No system packages are required on Windows.

Python packages
  - Linux:   fastapi, uvicorn, pydantic, pywebview, PyGObject, pycairo
             (requirements.txt)
  - Windows: fastapi, uvicorn, pydantic, pywebview[qt]
             (requirements-windows.txt)


## Installation
------------

Linux
  ./install.sh

Windows
  install.bat      (make sure Python is on PATH and installed)

Both scripts create `venv/`, upgrade pip, and install the Python packages.


## Running
-------

Linux
  ./llama-launcher.sh     (cd into the folder, activate venv, run app.py)
  Or manually:
    source venv/bin/activate
    python app.py

Windows
  llama-launcher.bat      (or double-click it)
  Or manually:
    venv\Scripts\activate
    python app.py

The app finds a free localhost port, starts the FastAPI server, then opens the
window at http://127.0.0.1:<port>.


## Configuration
-------------
Input your paths for llama.cpp and ik_llama and your model folders. Do delete the provided examples. You can do that in the gui or manually in the config.json file.

  - model_dirs: list of directories scanned for .gguf models.
  - backends: per-backend settings. Each entry has:
      bin:  absolute path to the backend build's `bin` directory
            (e.g. .../llama.cpp/build/bin), and
      port: the TCP port that backend should listen on.


presets.json
  Named sampling presets matched to models by glob pattern. params may include
  temp, topk, topp, presence_penalty, repeat_penalty, reasoning_budget,
  reasoning_effort, ctx, mem_mgmt, and per-flag forms such as
  {"temp": {"on": true, "v": 1}}.

overrides.json   (created at runtime)
  Manual per-model companion paths (mmproj_path, mtp_path, dflash_path) that
  survive rescans.


### Companion-file conventions
--------------------------
  The scanner groups GGUF shards (matching -NNNNN-of-MMMMM) and recognises
  companion files named with an mmproj / mtp / dflash prefix or suffix,
  optionally followed by quant/precision tags (e.g. q8_0, f16, bf16). The
  first shard becomes the launch file; sizes are summed across shards.


### Files
-----
  app.py                  Entry point: runs the server + opens the window.
  launcher.py             FastAPI backend + model scanning + config handling.
  gguf_read.py            GGUF header parser (metadata used for estimates).
  index.html              The web UI served by the backend.
  install.sh / install.bat          Installers (Linux / Windows).
  llama-launcher.sh / .bat          Run scripts (Linux / Windows).
  requirements.txt / requirements-windows.txt
  config.json, presets.json         Runtime configuration (edit for your setup).

## Contributing

Feel free to fork or add PRs if you like.

## License

This project is licensed under [MIT](LICENSE).

## Support

If you find this useful, consider buying me a coffee ☕

<a href="https://ko-fi.com/juanml82"> <img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Buy me a coffee on Ko-fi" /> </a>
