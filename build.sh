#!/bin/bash
set -e

echo "Installing MeCab system packages..."
apt-get update
apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

echo "Installing Python packages..."
pip install -r requirements.txt

echo "Downloading dictionaries..."
python -c "import unidic_lite; unidic_lite.download()" 2>/dev/null || true

echo "ULTIMATE SYSTEM READY!"
