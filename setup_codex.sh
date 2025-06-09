#!/usr/bin/env bash
# =============================================
#  setup_codex.sh – bootstrap deps in Codex
# =============================================
set -euo pipefail

echo "🔄  Upgrading pip…"
python -m pip install --quiet --upgrade pip

echo "📦  Installing project requirements…"
python -m pip install --quiet -r requirements.txt

echo -e "\n✅  All dependencies installed – happy hacking!"