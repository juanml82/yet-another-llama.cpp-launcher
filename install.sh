#!/usr/bin/env bash
set -euo pipefail

# ── 1. Runtime system dependencies (GTK / WebKit for the pywebview window) ──
sudo apt-get update -y
sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-webkit2-4.1

# ── 2. Compile dependencies (needed because PyGObject is pip-installed in the venv) ──
sudo apt-get install -y libgirepository1.0-dev libgirepository-2.0-dev \
                        libcairo2-dev pkg-config python3-dev

# ── 3. Virtual environment + Python packages (single source of truth: requirements.txt) ──
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
